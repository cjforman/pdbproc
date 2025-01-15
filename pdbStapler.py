#!/usr/bin/env python
import sys, os
import numpy as np
import pdbLib as pdb
import copy as cp
import itertools
import munkres

rot_epsilon = 1e-6

def isEqual(p1, p2, epsilon):
    # checks to see if two vectors are equal. 
    # Zero vector defined as having a magnitude less than epsilon. 
    equal = True # assume vectors are equal
    n = p1 - p2
    nMag =  np.linalg.norm(n)
    
    if ( abs(nMag - 0.0) > epsilon):
        equal = False
        
    return equal


def rotPAboutAxis(p, n, angle):
    # angle in radians
    # rotates the vector P about an axis n by an angle.

    nHat = n/np.linalg.norm(n) # ensure we are using a normalised vector
    try:
        outVec = p*np.cos(angle) + np.cross(nHat, p)*np.sin(angle) + nHat*np.dot(nHat, p)*(1- np.cos(angle))
    except ValueError:
        print("valueError")
        outVec = p

    return outVec


def rotPAboutAxisAtPoint(p, p0, n, angle):
    # angle in radians
    # rotates the vector P about an axis n through the point p0 by an angle.
    if not isEqual(p, p0, 1e-10):
        r = p - p0
        nHat = n/np.linalg.norm(n) # ensure we are using a normalised vector
        try:
            r_Rot = r*np.cos(angle) + np.cross(nHat, r)*np.sin(angle) + nHat*np.dot(nHat, r)*(1- np.cos(angle))
            outVec = r_Rot + p0
        except ValueError:
            print("valueError")
            outVec = p
    else:
        outVec = p
    return outVec

def _find_permutations(*args, **kwargs):
    return find_permutations_munkres(*args, **kwargs)
 
def permuteArray(Xold, perm):
    # don't modify Xold
    Xnew = np.copy(Xold)
    permsorted = sorted(perm)
    for (iold, inew) in itertools.izip(permsorted, perm):
        Xnew[inew*3:inew*3+3] = Xold[iold*3:iold*3+3]
 
    return Xnew
 
def _make_cost_matrix(X1, X2):
    """
    return the cost matrix for use in the hungarian algorithm.
     
    the cost matrix is the distance matrix (squared) for all atoms in atomlist
    """
    cost = (((X1[np.newaxis,:] - X2[:,np.newaxis,:])**2).sum(2))
    return cost
 
def find_permutations_munkres( X1, X2, make_cost_matrix=_make_cost_matrix ):
    """
    For a given set of positions X1 and X2, find the best permutation of the
    atoms in X2.
     
    The positions must already be reshaped to reflect the dimensionality of the system!
 
    Use an implementation of the Hungarian Algorithm in the Python package
    index (PyPi) called munkres (another name for the algorithm).  The
    hungarian algorithm time scales as O(n^3), much faster than the O(n!) from
    looping through all permutations.
 
    http://en.wikipedia.org/wiki/Hungarian_algorithm
    http://pypi.python.org/pypi/munkres/1.0.5.2
     
    another package, hungarian, implements the same routine in comiled C
    http://pypi.python.org/pypi/hungarian/
    When I first downloaded this package I got segfaults.  The problem for me
    was casing an integer pointer as (npy_intp *).  I may add the corrected 
    version to pele at some point
    """
    #########################################
    # create the cost matrix
    # cost[j,i] = (X1(i,:) - X2(j,:))**2
    #########################################
    cost = make_cost_matrix(X1, X2)
 
    #########################################
    # run the munkres algorithm
    #########################################
    matrix = cost.tolist()
    m = munkres.Munkres()
    newind = m.compute(matrix)    
     
    #########################################
    # apply the permutation
    #########################################
    costnew = 0.
    new_indices = range(len(X1))
    for (iold, inew) in newind:
        costnew += cost[iold, inew]
        new_indices[inew] = iold
 
    dist = np.sqrt(costnew)
    return dist, new_indices
 
def find_best_permutation(X1, X2, permlist=None, user_algorithm=None, 
                             reshape=True, user_cost_matrix=_make_cost_matrix,
                             **kwargs):
    """
    find the permutation of the atoms which minimizes the distance |X1-X2|
     
    With all the default parameters, findBestPermutation assumes that X1, X2
    are arrays of atoms in 3d space and performs reshaping on the coordinates. However,
    if you want to pass a 2D system or a custom array with own cost function, you can turn
    automatic reshaping off. 
     
    Parameters
    ----------
    X1, X2 : 
        the structures to align
    permlist : a list of lists
        A list of lists of atoms which are interchangable.
        e.g. for a 50/50 binary mixture::
         
            permlist = [range(1,natoms/2), range(natoms/2,natoms)]
         
        If permlist is None all atoms are assumed to be permutable.
 
    user_algoriithm : None or callable
        you can optionally pass which algorithm to use.
    gen_cost_matrix : None or callable
        user function to generate the cost matrix
    reshape : boolean
        shall coordinate reshaping be performed.
    box_lengths : float array
        array of floats giving the box lengths for periodic boundary conditions.
        Set to None for no periodic boundary conditions.
     
    Returns
    -------
    dist : float
        the minimum distance WARNING: THIS IS NOT NECESSARILY CORRECT, IT SHOULD BE 
        RECALCULATED.  THIS WILL BE REMOVED IN THE FUTURE.
    perm:
        a permutation which will best align coords2 with coords1
     
    Notes
    -----
    For each list of interchangeable atoms in permlist the permutation
    which minimizes the distance between the two structures is found.  This minimimization
    is done by mapping the problem onto the linear assignment problem which can then be solved
    using graph theoretic techniques.  
     
    http://en.wikipedia.org/wiki/Linear_assignment_problem
    http://en.wikipedia.org/wiki/Hungarian_algorithm
 
    there are several packages in pypi which solve the linear assignment problem
     
    hungarian : c++ code wrapped in python.  scales roughly like natoms**2.5
     
    munkres : completely in python. scales roughly like natoms**3.  very slow for natoms > 10
     
    in addition we have wrapped the OPTIM version for use in pele.  It uses the sparse 
    version of the Jonker-Volgenant algorithm.  Furthermore the cost matrix calculated in 
    a compiled language for an additional speed boost. It scales roughly like natoms**2
 
    """
    if reshape:
        X1 = X1.reshape([-1,3])
        X2 = X2.reshape([-1,3])
     
    if permlist is None:
        permlist = [range(len(X1))]
     
    newperm = range(len(X1))
    disttot = 0.
     
    for atomlist in permlist:
        if len(atomlist) == 0:
            continue
        if user_algorithm is not None:
            dist, perm = user_algorithm(X1[atomlist], X2[atomlist], make_cost_matrix=user_cost_matrix, **kwargs)
        else:
            dist, perm = _find_permutations(X1[atomlist], X2[atomlist], **kwargs)
         
        disttot += dist**2
        for atom, i in zip(atomlist,xrange(len(atomlist))):
            newperm[atom] = atomlist[perm[i]]
    dist = np.sqrt(disttot)
    return dist, newperm
 
def _cartesian_distance_periodic(x1, x2, box_lengths):
    dim = len(box_lengths)
    dx = x2 - x1
    dx = dx.reshape([-1,dim])
    dx -= box_lengths * np.round(dx / box_lengths[np.newaxis, :])
    dx = dx.ravel()
    dist = np.linalg.norm(dx)
    return dist
 
def _cartesian_distance(x1, x2, box_lengths=None):
    if box_lengths is None:
        return np.linalg.norm(x2-x1)
    else:
        return _cartesian_distance_periodic(x1, x2, box_lengths)
 
def optimize_permutations(X1, X2, permlist=None, user_algorithm=None,
                           recalculate_distance=_cartesian_distance,
                           box_lengths=None,
                           **kwargs):
    """return the best alignment of the structures X1 and X2 after optimizing permutations
     
    Parameters
    ----------
    X1, X2 : 
        the structures to align.  X1 will be left unchanged.
    permlist : a list of lists
        A list of lists of atoms which are interchangable.
        e.g. for a 50/50 binary mixture::
         
            permlist = [range(1,natoms/2), range(natoms/2,natoms)]
 
    user_algoriithm : None or callable
        you can optionally pass which algorithm to use to optimize the permutations the structures
    gen_cost_matrix : None or callable
        user function to generate the cost matrix
    recalculate_distance : callable
        function to compute the distance of the optimized coords.  If None is passed
        then the distance is not recalculated and the returned distance is unreliable.
    reshape : boolean
        shall coordinate reshaping be performed.
    box_lengths : float array
        array of floats giving the box lengths for periodic boundary conditions.
        Set to None for no periodic boundary conditions.
     
    Returns
    -------
    dist : float
        the minimum distance
    X1, X2new:
        the optimized coordinates
 
    See Also
    --------
    find_best_permutation : 
        use this function to find the optimized permutation without changing the coordinates.
    """
    if box_lengths is not None:
        kwargs["box_lengths"] = box_lengths
    dist, perm = find_best_permutation(X1, X2, permlist=permlist, 
                    user_algorithm=user_algorithm, **kwargs)
    X2_ = X2.reshape([-1, 3])
    X2new = X2_[perm].flatten()
     
    if recalculate_distance is not None:
        # Recalculate the distance.  We can't trust the returned value
        dist = _cartesian_distance(X1, X2new, box_lengths)
     
    return dist, X1, X2new
 
 
def findrotation_kabsch(coords1, coords2, align_com=True):
    """
    Kabsch, Wolfgang, (1976) "A solution of the best rotation to relate two sets of vectors", Acta Crystallographica 32:922
     
    ..note::
        this has different return values than findrotation_kearsley.  The return values for this
        function may change in the future.
    """
    # check if arrays are of same size
    if coords1.size != coords2.size:
        raise ValueError("dimension of arrays does not match")
     
    # reshape the arrays
    x1 = coords1.reshape([-1,3]).copy()
    x2 = coords2.reshape([-1,3]).copy()
     
    # determine number of atoms
    natoms = x1.shape[0]
     
    # set both com to zero
    if align_com:
        com1 = np.sum(x1, axis=0) / float(natoms)
        com2 = np.sum(x2, axis=0) / float(natoms)
        x1 -= com1
        x2 -= com2
   
    # calculate covariance matrix
    A = np.dot( x2.transpose(), x1)
    # and do single value decomposition
    u, s, v = np.linalg.svd(A)
  
    if np.linalg.det(u) * np.linalg.det(v) + 1.0 < 1e-8:
        s[-1] = -s[-1]
        u[:,-1] = -u[:,-1]
 
    return  np.dot(u, v).transpose()
     
def findrotation_kearsley(x1, x2, align_com=True):
    """Return the rotation matrix which aligns XB with XA
     
    Return the matrix which
    aligns structure XB to be as similar as possible to structure XA.
    To be precise, rotate XB, so as to minimize the distance |XA - XB|.
 
    Rotations will be done around the origin, not the center of mass
 
    Rotational alignment follows the prescription of
    Kearsley, Acta Cryst. A, 45, 208-210, 1989
    http://dx.doi.org/10.1107/S0108767388010128
    """
    if x1.size != x2.size:
        raise ValueError("dimension of arrays does not match")
     
    # reshape the arrays
    x1 = x1.reshape([-1,3]).copy()
    x2 = x2.reshape([-1,3]).copy()
    # determine number of atoms
    natoms = x1.shape[0]
     
    # set both com to zero
    if align_com:
        com1 = np.sum(x1,axis=0) / float(natoms)
        com2 = np.sum(x2,axis=0) / float(natoms)
        x1 -= com1
        x2 -= com2
 
    x1 = x1.ravel() 
    x2 = x2.ravel()
     
    # TODO: this is very dirty!
    #########################################
    # Create matrix QMAT
    #########################################
 
    QMAT = np.zeros([4,4], np.float64)
    for J1 in range(natoms):
        J2 = 3* J1 -1
        XM = x1[J2+1] - x2[J2+1]
        YM = x1[J2+2] - x2[J2+2]
        ZM = x1[J2+3] - x2[J2+3]
        XP = x1[J2+1] + x2[J2+1]
        YP = x1[J2+2] + x2[J2+2]
        ZP = x1[J2+3] + x2[J2+3]
        QMAT[0,0] = QMAT[0,0] + XM**2 + YM**2 + ZM**2
        QMAT[0,1] = QMAT[0,1] - YP*ZM + YM*ZP
        QMAT[0,2] = QMAT[0,2] - XM*ZP + XP*ZM
        QMAT[0,3] = QMAT[0,3] - XP*YM + XM*YP
        QMAT[1,1] = QMAT[1,1] + YP**2 + ZP**2 + XM**2
        QMAT[1,2] = QMAT[1,2] + XM*YM - XP*YP
        QMAT[1,3] = QMAT[1,3] + XM*ZM - XP*ZP
        QMAT[2,2] = QMAT[2,2] + XP**2 + ZP**2 + YM**2
        QMAT[2,3] = QMAT[2,3] + YM*ZM - YP*ZP
        QMAT[3,3] = QMAT[3,3] + XP**2 + YP**2 + ZM**2
 
    QMAT[1,0] = QMAT[0,1]
    QMAT[2,0] = QMAT[0,2]
    QMAT[2,1] = QMAT[1,2]
    QMAT[3,0] = QMAT[0,3]
    QMAT[3,1] = QMAT[1,3]
    QMAT[3,2] = QMAT[2,3]
 
    ###########################################
    """
    Find eigenvalues and eigenvectors of QMAT.  The eigenvector corresponding
    to the smallest eigenvalue is the quaternion which rotates XB into best
    alignment with XA.  The smallest eigenvalue is the squared distance between
    the resulting structures.
    """
    ###########################################
    (eigs, vecs) = np.linalg.eig(QMAT)
 
    imin = np.argmin(eigs)
    eigmin = eigs[imin] # the minimum eigenvector
    Q2 = vecs[:,imin]  # the eigenvector corresponding to the minimum eigenvalue
    if eigmin < 0.:
        if abs(eigmin) < 1e-6:
            eigmin = 0.
        else:
            print('minDist> WARNING minimum eigenvalue is ', eigmin, ' change to absolute value')
            eigmin = -eigmin
 
    dist = np.sqrt(eigmin) # this is the minimized distance between the two structures
 
    Q2 = np.real_if_close(Q2, 1e-10)
    if np.iscomplexobj(Q2):
        raise ValueError("Q2 is complex")
    return dist, q2mx(Q2)
 
findrotation = findrotation_kearsley
 
class StandardClusterAlignment(object):
    """
    class to iterate over standard alignments for atomic clusters
 
    Quickly determines alignments of clusters which are possible exact matches.
    It uses atoms which are far away from the center to determine possible
    rotations. The algorithm does the following:
 
    1) Get 2 reference atoms from structure 1 which are farthest away from center
       and are not linear
    2) Determine candidates from structure 2 which are in same shell
       as reference atoms from structure 1 (+- accuracy)
    3) loop over all candidate combinations to determine
       orientation and check for match. Skip directly if angle of candidates
       does not match angle of reference atoms in structure 1.
 
    Parameters
    ----------
    coords1 : np.array
        first coordinates
    coords2 : np.array
        second coordinates
    accuracy : float
        accuracy of shell for atom candidates in standard alignment
    can_invert : boolean
        is an inversion possible?
 
    Examples
    --------
 
    >> for rot, invert in StandardClusterAlignment(X1, X2):
    >>     print( "possible rotation:",rot,"inversion:",invert
 
    """
    def __init__(self, coords1, coords2, accuracy = 0.01, can_invert=True):
        x1 = coords1.reshape([-1,3]).copy()
        x2 = coords2.reshape([-1,3]).copy()
 
        self.accuracy = accuracy
        self.can_invert = can_invert
 
        # calculate distance of all atoms
        R1 = np.sqrt(np.sum(x1*x1, axis=1))
        R2 = np.sqrt(np.sum(x2*x2, axis=1))
 
        # at least 2 atoms are needed
        # get atom most outer atom
 
        # get 1. reference atom in configuration 1
        # use the atom with biggest distance to com
        idx_sorted = R1.argsort()
        idx1_1 = idx_sorted[-1]
 
        # find second atom which is not in a line
        cos_best = 99.00
        for idx1_2 in reversed(idx_sorted[0:-1]):
            # stop if angle is larger than threshold
            cos_theta1 = np.dot(x1[idx1_1], x1[idx1_2]) / \
                (np.linalg.norm(x1[idx1_1])*np.linalg.norm(x1[idx1_2]))
 
            # store the best match in case it is a almost linear molecule
            if np.abs(cos_theta1) < np.abs(cos_best):
                cos_best = cos_theta1
                idx1_2_best = idx1_2
 
            if np.abs(cos_theta1) < 0.9:
                break
 
        idx1_2 = idx1_2_best
 
        # do a very quick check if most distant atom from
        # center are within accuracy
        if np.abs(R1[idx1_1] - R2.max()) > accuracy:
            candidates1 = []
            candidates2 = []
        else:
            # get indices of atoms in shell of thickness 2*accuracy
            candidates1 = np.arange(len(R2))[ \
                 (R2 > R1[idx1_1] - accuracy)*(R2 < R1[idx1_1] + accuracy)]
            candidates2 = np.arange(len(R2))[ \
                 (R2 > R1[idx1_2] - accuracy)*(R2 < R1[idx1_2] + accuracy)]
 
        self.x1 = x1
        self.x2 = x2
        self.idx1_1 = idx1_1
        self.idx1_2 = idx1_2
        self.idx2_1 = None
        self.idx2_2 = None
        self.invert = False
 
        self.cos_theta1 = cos_theta1
        self.candidates2 = candidates2
 
        self.iter1 = iter(candidates1)
        self.iter2 = iter(self.candidates2)
 
    def __iter__(self):
        return self
 
    def __next__(self):
        # obtain first index for first call
        if self.idx2_1 is None:
            self.idx2_1 = self.iter1.__next__()
 
        # toggle inversion if inversion is possible
        if self.can_invert and self.invert == False and self.idx2_2 is not None:
            self.invert = True
        else:
            # determine next pair of indices
            self.invert = False
            # try to increment 2nd iterator
            try:
                self.idx2_2 = self.iter2.next()
            except StopIteration:
                # end of list, start over again
                self.iter2 = iter(self.candidates2)
                # and increment iter1
                self.idx2_1 = self.iter1.next()
                self.idx2_2 = None
                return self.next()
 
        if self.idx2_1 == self.idx2_2:
            return self.next()
 
        x1 = self.x1
        x2 = self.x2
        idx1_1 = self.idx1_1
        idx1_2 = self.idx1_2
        idx2_1 = self.idx2_1
        idx2_2 = self.idx2_2
 
        assert idx1_1 is not None
        assert idx1_2 is not None
        assert idx2_1 is not None
        assert idx2_2 is not None
 
        # we can immediately trash the match if angle does not match
        try:
            cos_theta2 = np.dot(x2[idx2_1], x2[idx2_2]) / \
                (np.linalg.norm(x2[idx2_1])*np.linalg.norm(x2[idx2_2]))
        except ValueError:
            raise
        if np.abs(cos_theta2 - self.cos_theta1) > 0.5:
            return self.next()
 
        mul = 1.0
        if self.invert:
            mul=-1.0
 
        # get rotation for current atom match candidates
        dist, rot = findrotation(
            x1[[idx1_1, idx1_2]], mul*x2[[idx2_1, idx2_2]], align_com=False)
 
        return rot, self.invert
 
 
 
class TransformPolicy(object):
    """ interface for possible transformations on a set of coordinates
 
    The transform policy tells minpermdist how to perform transformations,
    i.e. a translation, rotation and inversion on a specific set of
    coordinates. This class is necessary since in general a coordinate array
    does not carry any information  on the type of coordinate, e.g. if it's a
    site coordinate, atom coordinate or angle axis vector.
 
    All transformation act in place, that means they change the current
    coordinates and do not make a copy.
 
    """
      
    def translate(self, X, d):
        """ translate the coordinates """
        raise NotImplementedError
     
    def rotate(self, X, mx):
        """ apply rotation matrix mx for a rotation around the origin"""
        raise NotImplementedError
     
    def can_invert(self):
        """ returns True or False if an inversion can be performed"""
        raise NotImplementedError
     
    def invert(self, X):
        """ perform an inversion at the origin """
        raise NotImplementedError
     
    def permute(self, X, perm):
        """ returns the permuted coordinates """
     
class MeasurePolicy(object):
    """ interface for possible measurements on a set of coordinates
 
    The MeasurePolicy defines an interface which defines how to perform
    certain measures which are essential for minpermdist on a set of
    coordinates. For more motivation of this class see TransformPolicy.
    """
     
    def get_com(self, X):
        """ calculate the center of mass """
        raise NotImplementedError
     
    def get_dist(self, X1, X2, with_vector=False):
        """ calculate the distance between 2 set of coordinates """
        raise NotImplementedError
     
    def find_permutation(self, X1, X2):
        """ find the best permutation between 2 sets of coordinates """
        raise NotImplementedError
     
    def find_rotation(self, X1, X2):
        """ find the best rotation matrix to bring structure 2 on 1 """
        raise NotImplementedError
 
class TransformAtomicCluster(TransformPolicy):
    """ transformation rules for atomic clusters """
     
    def __init__(self, can_invert=True):
        self._can_invert = can_invert
     
    @staticmethod
    def translate(X, d):
        Xtmp = X.reshape([-1,3])
        Xtmp += d
     
    @staticmethod
    def rotate(X, mx,):
        Xtmp = X.reshape([-1,3])
        Xtmp = np.dot(mx, Xtmp.transpose()).transpose()
        X[:] = Xtmp.reshape(X.shape)
     
    @staticmethod        
    def permute(X, perm):
        a = X.reshape(-1,3)[perm].flatten()
        # now modify the passed object, X
        X[:] = a[:]
        return X
         
    def can_invert(self):
        return self._can_invert
     
    @staticmethod
    def invert(X):
        X[:] = -X
         
class MeasureAtomicCluster(MeasurePolicy):
    """ measure rules for atomic clusters """
     
    def __init__(self, permlist=None):
        self.permlist = permlist
     
    def get_com(self, X):
        X = np.reshape(X, [-1,3])
        natoms = len(X[:,0])
        com = X.sum(0) / natoms
        return com
 
    def get_dist(self, X1, X2, with_vector=False):
        dist = np.linalg.norm(X1.ravel()-X2.ravel())
        if with_vector:
            return dist, X2-X1
        else:
            return dist
     
    def find_permutation(self, X1, X2):
        return find_best_permutation(X1, X2, self.permlist)
     
    def find_rotation(self, X1, X2):
        dist, mx = findrotation(X1, X2)
        return dist, mx
     
class MinPermDistCluster(object):
    """
    Minimize the distance between two clusters.  
     
    Parameters
    ----------
    niter : int
        the number of basinhopping iterations to perform
    verbose : boolean 
        whether to print status information
    accuracy :float, optional
        accuracy for standard alignment which determines if the structures are identical
    tol : float, optional
        tolerance for an exact match to stop iterations
    transform : 
        Transform policy which tells MinpermDist how to transform the given coordinates
    measure : 
        measure policy which tells minpermdist how to perform certains measures on the coordinates.
     
    Notes
    -----
 
    The following symmetries will be accounted for::
     
    1. Translational symmetry
    #. Global rotational symmetry
    #. Permutational symmetry
    #. Point inversion symmetry
 
     
    The algorithm here to find the best distance is
     
    for rotation in standardalignments:
        optimize permutation
        optimize rotation
        check_match
         
    for i in range(niter):    
        random_rotation
        optimize permutations
        align rotation
        check_match
         
    The minpermdist algorithm is generic and can act on various types of
    coordinates, e.g. carthesian, angle axis, .... The transform and measure
    policies define and interface to manipulate and analyze a given set of
    coordinates. If the coordinates don't have a standard format, custom policies
    can be specified. As an example see the angle axis minpermdist routines.
         
    See also
    --------
    TransformPolicy, MeasurePolicy
     
    """
    def __init__(self, niter=10, verbose=False, tol=0.01, accuracy=0.01,
                 measure=MeasureAtomicCluster(), transform=TransformAtomicCluster()):
         
        self.niter = niter
         
        self.verbose = verbose
        self.measure = measure
        self.transform=transform
        self.accuracy = accuracy
        self.tol = tol
         
    def check_match(self, x1, x2, rot, invert):
        """ check a given rotation for a match """
        x2_trial = x2.copy()
        if invert:
            self.transform.invert(x2_trial)
        self.transform.rotate(x2_trial, rot)
 
 
        # get the best permutation
        dist, perm = self.measure.find_permutation(x1, x2_trial)
        x2_trial = self.transform.permute(x2_trial, perm)
        
        # now find best rotational alignment, this is more reliable than just
        # aligning the 2 reference atoms
        dist, rot2 = self.measure.find_rotation(x1, x2_trial)
        self.transform.rotate(x2_trial, rot2)
        # use the maximum distance, not rms as cutoff criterion
         
        dist =  self.measure.get_dist(x1, x2_trial)
         
        if dist < self.distbest:
            self.distbest = dist
            self.rotbest = np.dot(rot2, rot)
            self.invbest = invert
            self.x2_best = x2_trial    
     
    def finalize_best_match(self, x1):
        """ do final processing of the best match """
        self.transform.translate(self.x2_best, self.com_shift)
        dist = self.measure.get_dist(x1, self.x2_best)
        if np.abs(dist - self.distbest) > 1e-6:
            raise RuntimeError        
        if self.verbose:
            print( "finaldist", dist, "distmin", self.distbest )
 
        return dist, self.x2_best
 
    def _standard_alignments(self, x1, x2):
        """ get iterator for standard alignments """
        return StandardClusterAlignment(x1, x2, accuracy=self.accuracy, 
                                        can_invert=self.transform.can_invert())  
        
    def align_structures(self, coords1, coords2):        
        """
        Parameters
        ----------
        coords1, coords2 : np.array
            the structures to align.  X2 will be aligned with X1, both
            the center of masses will be shifted to the origin
 
        Returns
        -------
        a triple of (dist, coords1, coords2). coords1 are the unchanged coords1
        and coords2 are brought in best alignment with coords2
        """
 
        # we don't want to change the given coordinates
        coords1 = coords1.copy()
        coords2 = coords2.copy()
         
        x1 = np.copy(coords1)
        x2 = np.copy(coords2)
 
        com1 = self.measure.get_com(x1)
        self.transform.translate(x1, -com1)
        com2 = self.measure.get_com(x2)
        self.transform.translate(x2, -com2)
 
        self.com_shift = com1
         
        self.mxbest = np.identity(3)
        self.distbest = self.measure.get_dist(x1, x2)
        self.x2_best = x2.copy()
         
        # sn402: The unlikely event that the structures are already nearly perfectly aligned.
        if self.distbest < self.tol:
            dist, x2 = self.finalize_best_match(coords1)
            return self.distbest, coords1, x2
         
        for rot, invert in self._standard_alignments(x1, x2):
            self.check_match(x1, x2, rot, invert)
            if self.distbest < self.tol:
                dist, x2 = self.finalize_best_match(coords1)
                return dist, coords1, x2
         
        # if we didn't find a perfect match here, try random rotations to optimize the match
        for i in range(self.niter):
            rot = aa2mx(random_aa())
            self.check_match(x1, x2, rot, False)
            if self.transform.can_invert():
                self.check_match(x1, x2, rot, True)
 
        # TODO: should we do an additional sanity check for permutation / rotation?        
         
        dist, x2 = self.finalize_best_match(coords1)
         
        return dist, coords1, x2
     
    def __call__(self, coords1, coords2):
        return self.align_structures(coords1, coords2)
 
 
class MinPermDistAtomicCluster(MinPermDistCluster):
    """ minpermdist for atomic cluster (3 carthesian coordinates per site)
 
    Parameters
    ----------
 
    permlist : optional
        list of allowed permutations. If nothing is given, all atoms will be
        considered as permutable. For no permutations give an empty list []
    can_invert : bool, optional
        also test for inversion
 
    See also
    --------
 
    MinPermDistCluster
 
    """
    def __init__(self, permlist=None, can_invert=True, **kwargs):
        transform=TransformAtomicCluster(can_invert=can_invert)
        measure = MeasureAtomicCluster(permlist=permlist)
         
        MinPermDistCluster.__init__(self, transform=transform, measure=measure, **kwargs)

def q2aa(qin):
    """
    quaternion to angle axis
    
    Parameters
    ----------
    Q: quaternion of length 4
    
    Returns
    -------
    output V: angle axis vector of lenth 3
    """
    q = np.copy(qin)
    if q[0] < 0.: q = -q
    if q[0] > 1.0: q /= np.sqrt(np.dot(q, q))
    theta = 2. * np.arccos(q[0])
    s = np.sqrt(1. - q[0] * q[0])
    if s < rot_epsilon:
        p = 2. * q[1:4]
    else:
        p = q[1:4] / s * theta
    return p

def random_aa():
    """return a uniformly distributed random angle axis vector"""
    return q2aa(random_q())


def aa2mx(aa):
    a = np.linalg.norm(aa)
    n = aa/a
    x=n[0]
    y=n[1]
    z=n[2]
    c = np.cos(a)
    s = np.sin(a)
    t = 1 - c
    
    return np.array([ [ t * x * x + c,     t * x * y - z * s,  t * x * z + y * s ],
                      [ t * x * y + z * s, t * y * y + c    ,  t * y * z - x * s ],
                      [ t * x * z - y * s, t * y * z + x * s,  t * z * z + c     ]])

def random_q():
    """
    uniform random rotation in angle axis formulation
    
    Notes
    -----
    input: 3 uniformly distributed random numbers
    uses the algorithm given in
    K. Shoemake, Uniform random rotations, Graphics Gems III, pages 124-132. Academic, New York, 1992.
    This first generates a random rotation in quaternion representation. We should substitute this by
    a direct angle axis generation, but be careful: the angle of rotation in angle axis representation
    is NOT uniformly distributed
    """
    from numpy import sqrt, sin, cos, pi

    u = np.random.uniform(0, 1, [3])
    q = np.zeros(4, np.float64)
    q[0] = sqrt(1. - u[0]) * sin(2. * pi * u[1])
    q[1] = sqrt(1. - u[0]) * cos(2. * pi * u[1])
    q[2] = sqrt(u[0]) * sin(2. * pi * u[2])
    q[3] = sqrt(u[0]) * cos(2. * pi * u[2])
    return q

def q2mx(qin):
    """quaternion to rotation matrix"""
    Q = qin / np.linalg.norm(qin)
    RMX = np.zeros([3, 3], np.float64)
    Q2Q3 = Q[1] * Q[2]
    Q1Q4 = Q[0] * Q[3]
    Q2Q4 = Q[1] * Q[3]
    Q1Q3 = Q[0] * Q[2]
    Q3Q4 = Q[2] * Q[3]
    Q1Q2 = Q[0] * Q[1]

    RMX[0, 0] = 2. * (0.5 - Q[2] * Q[2] - Q[3] * Q[3])
    RMX[1, 1] = 2. * (0.5 - Q[1] * Q[1] - Q[3] * Q[3])
    RMX[2, 2] = 2. * (0.5 - Q[1] * Q[1] - Q[2] * Q[2])
    RMX[0, 1] = 2. * (Q2Q3 - Q1Q4)
    RMX[1, 0] = 2. * (Q2Q3 + Q1Q4)
    RMX[0, 2] = 2. * (Q2Q4 + Q1Q3)
    RMX[2, 0] = 2. * (Q2Q4 - Q1Q3)
    RMX[1, 2] = 2. * (Q3Q4 - Q1Q2)
    RMX[2, 1] = 2. * (Q3Q4 + Q1Q2)
    return RMX

class chain:
    """a class for storing information about a chain"""
    def __init__(self, chainName, startResidue, endResidue):
        self.chainName = chainName
        self.startResidue = startResidue
        self.endResidue = endResidue
        self.chainLength = endResidue - startResidue + 1

    def Display(self):
        print( self.chainName )
        print( self.startResidue )
        print( self.endResidue )
        print( self.chainLength )
        return

    # assumes residue numbers in a chain are monotonic and contiguous
    def checkResidueInChain(self, resNum):
        return (resNum >= self.startResidue) and (resNum <= self.endResidue)

    def getChainName(self):
        return self.chainName
 
    def getChainLength(self):
        return self.chainLength

class Fragment:
    """A class for storing information pertaining to a fragment"""
    def __init__(self, chains, preResList, postResList, NCapList, CCapList, filename):
        self.chains = chains
        self.chainNames = []
        for chain in self.chains:
            self.chainNames.append(chain.getChainName())
        self.name = filename
        self.atoms = pdb.readAtoms(filename)
        self.preResList = cp.copy(preResList)
        self.postResList = cp.copy(postResList)
        self.NCapList = cp.copy(NCapList)
        self.CCapList = cp.copy(CCapList)
 
    # debugging routine to check the contents of a fragment class
    def Display(self):
        print( "Chain Names:" )
        print( self.chainNames )
        print( "Chain Info:" )       
        for chain in self.chains:
            chain.Display()
        print( "Filename:" )
        print( self.name )
        print( "Number of atoms" )
        print( len(self.atoms) )
        print( "Residues Attaching to Previous Fragment" )
        print( self.preResList )
        print( "Residues Attaching to Next Fragment" )
        print( self.postResList )
        print( "NCap chains" )
        print( self.NCapList )
        print( "CCap chains" )
        print( self.CCapList )
        return
   
    # computes centre of gravity of entire fragment
    def COG(self):
        vecList = []
        for atom in self.atoms:
            vecList.append(np.array([atom[7], atom[8], atom[9]]))

        # compute centre of gravity of all vectors in list
        return sum(vecList) / len(vecList)

    def estimateNextResidueC(self,chainName):
        #get the atoms associated with the current chain
        atoms=self.extractChainAtoms(chainName)
        #loop through the atoms once and extract the particular atoms of interest near N terminus
        for atom in atoms:
            #get the atoms of interest for N terminus
            if atom[5]==atoms[0][5] and atom[1]=='N':
                N1=np.array([atom[7],atom[8],atom[9]])
            if atom[5]==atoms[0][5] and atom[1]=='C':
                C1=np.array([atom[7],atom[8],atom[9]])
            if atom[5]==atoms[0][5]+1 and atom[1]=='N':
                N2=np.array([atom[7],atom[8],atom[9]])
        #estimate the position of the C in the next residue. (current N terminus + vector from N to C in last residue)
        return N1 + C1 - N2

    def estimateNextResidueN(self,chainName):
        #get the atoms associated with the current chain
        atoms=self.extractChainAtoms(chainName)
        #loop through the atoms once and extract the particular atoms of interest near C terminus
        for atom in atoms:
            #get the atoms of interest for C terminus
            if atom[5]==atoms[-1][5]-1 and atom[1]=='C':
                CM1=np.array([atom[7],atom[8],atom[9]])
            if atom[5]==atoms[-1][5] and atom[1]=='N':
                NM0=np.array([atom[7],atom[8],atom[9]])
            if atom[5]==atoms[-1][5] and atom[1]=='C':
                CM0=np.array([atom[7],atom[8],atom[9]])
            
        #estimate the position of the N in the next residue. (current C terminus + vector from C to N in last residue)
        return CM0 + NM0 - CM1
        
    def checkNCap(self, chainName):
        return chainName in self.NCapList

    def checkCCap(self, chainName):
        return chainName in self.CCapList

    # returns a list of all atoms in a particular chain
    def extractChainAtoms(self, chainName):
        testChainName = chainName

        if chainName=='_':
            testChainName=' '
        
        return [atom for atom in self.atoms if atom[4]==testChainName]

    def getChainLength(self, chainName):
        for chain in self.chains:
            if (chain.getChainName() == chainName):
                length = chain.getChainLength()
        return length

    # returns a list of chain names in the current fragment
    def getChainNames(self):
        return self.chainNames

    # rewrite the xyz data in the atoms list with an input list of XYZ coordinates. Up to the user to preserve the original higher up.
    def updateXYZ(self, newXYZ):
        # check input and object data consistency
        if 3 * len(self.atoms) != len(newXYZ):
            print( 'Unable to update atomic XYZ; inconsistent number of atoms' )
            sys.exit(0)
        j = 0
        for atom in self.atoms:
            atom[7] = newXYZ[j]
            atom[8] = newXYZ[j + 1]
            atom[9] = newXYZ[j + 2]
            j = j + 3
        return 

    # extract all the xyz co-ordinates
    def extractAllXYZ(self):
 
        # extract xyz data as a list of ordinates x0,y0,z0,x1,y1,z1,...xn,yn,zn
        outList = []
        for atom in self.atoms:
            outList.append(atom[7])
            outList.append(atom[8])
            outList.append(atom[9])

        return np.array(outList)

    # extract the NCC coordinates of the atoms to align relative to their COG
    # return in the format of a list of floats x0, y0, z0, x1, y1, z1, ..., xn, yn, zn
    # rather than the numpy array([format])
    def extractXYZ(self, direction):
 
        # determine which sub set of linking residues we need the pre or the post set          
        if direction == 'post':
            resList = self.postResList
        if direction == 'pre':
            resList = self.preResList
            
        # extract the appropriate N, CA, C co-ordinates and store them in the order they appear in the atoms array
        vecList = [ np.array([atom[7], atom[8], atom[9]]) for atom in self.atoms if (str(atom[5]) in resList) and (atom[1] in ['N','CA','C'])] 

        # compute centre of gravity of all vectors in list
        COG = sum(vecList) / len(vecList)
 
        # recompute positions relative to COG.
        vecListC = [ vec - COG for vec in vecList]

        # convert to standard GMIN format
        outList = []
        for vec in vecListC:
            outList.append(vec[0])
            outList.append(vec[1])
            outList.append(vec[2])

        return COG, np.array(outList)

# taking a filename for sys.arg, read in file and load the data
# check the data for consistency and return the input data
# which consists of a filename for output, a list of fragment objects and a list of staple objects.
def getInput():

    if len(sys.argv) < 2:  # the program name and the one argument
        # stop the program and print an error message
        sys.exit("pdbStapler.py <instructionFile> [PAG] [atomgroups <inputPDB>] [relax]")

    filename = sys.argv[1]

    # open the input file
    try:
        fH = open(filename, 'r')
    except:
        raise Exception( "Unable to open input file: " + filename )
        sys.exit(0)

    # read in the text defined in pdbLib.py
    instructions = pdb.readTextFile(filename)

    # parse the input instructions
    outfile, fragments, staples, rotations = parseInstructions(instructions)

    proceed, errorMsg = validateInputConsistency(fragments, staples)
    
    print( fragments )
    print( staples )
    

    if proceed == 1:
        print( 'Unable to continue; input inconsistent:' )
        print( errorMsg )
        sys.exit(0)

    fH.close()

    PAG = 0
    relax = 0
    atomgroups = 0
    if 'PAG' in sys.argv:
        PAG = 1

    if 'relax' in sys.argv:
        relax = 1

    AGFilename = []
    wordIndex = 0
    for word in sys.argv:
        if word == 'atomgroups':
            atomgroups = 1
            AGFilename = sys.argv[wordIndex + 1]
        wordIndex += 1

    return outfile, fragments, staples, rotations, [PAG, atomgroups, AGFilename, relax]

# function to parse an instructions file read in as text
def parseInstructions(instructions):

    # initialise output variables
    fragments = []
    staples = []
    outfile = []
    rotations = []

    #initialise blocktype - a state variable for controlling which list the output goes into
    blockType='None'

    # loop through the input data
    for line in instructions:

        if line == '\n':
            keyword = 'NULL'
            dataList = 'NULL'
        else:
            #tokenise the input line
            data=line.split()
            # separate out the parts of the instruction
            keyword = data[0]
            dataList = data[1:]

        # get the outputfile name
        if keyword == 'OUTPUTFILE':
            outfile = cp.copy(dataList[0])

        if keyword == "ROTATION":
            rotations.append( (int(cp.copy(dataList[0])), float(cp.copy(dataList[1]))) )

        # process a fragment block in the input file; use the same code to process fragments and staples
        if keyword in ['FRAGMENT', 'STAPLE']:
            # record the nature of the block (staple or fragment)
            blockType = keyword
            # clear the lists of dummy variables for this fragment
            chains = []
            preResList = []
            postResList = []
            NCap = 'None'
            CCap = 'None'

            # get file name containing the atomic data
            infile = cp.copy(dataList[0])

        #if the keyword is end then generate the object corresponding to the fragment/staple and add it to the output list.
        #Use whatever values the variables have at the moment.
        if keyword == 'END':
            # create a fragment object and append it to the list of the fragments or staples as appropriate
            # the fragment object constructor will load the atomic data from the file
            if blockType == 'FRAGMENT':
                fragments.append(Fragment(chains, preResList, postResList, NCap, CCap, infile))
            else:
                staples.append(Fragment(chains, preResList, postResList, NCap, CCap, infile))
    
        #for the remaining keyword types just process each keyword statement accordingly and add the data to the various variables       
        if keyword == 'CHAIN':
            chains.append(chain(dataList[0], int(dataList[1]), int(dataList[2])))
    
        if keyword == 'PREV':
            preResList = cp.copy(dataList)
    
        if keyword == 'NEXT':
            postResList = cp.copy(dataList)
    
        if keyword == 'NCAP':
            NCap = cp.copy(dataList)
    
        if keyword == 'CCAP':
            CCap = cp.copy(dataList)
                
    return outfile, fragments, staples, rotations

def validateInputConsistency(fragments, staples):

    numFrags = len(fragments)
    numStaples = len(staples)

    errorMsg = ''

    # initialise the proceed variable. Zero good. Assume success unless fail.
    proceed = 0
    if numFrags - numStaples != 1:
        proceed = 1
        errorMsg = 'Wrong number of staples or fragments. nF = nS + 1.'

    # could put other checks in here. But we'll find out soon enough if the data makes no sense.
    # no point in going through it twice. can add checks to crash gracefully throughout

    return proceed, errorMsg

# takes a static fragment and a mobile fragment and aligns 
# the mobile one to the static one.
# the information for which residues are to be used for the alignment 
# is held within each structure
# returns an entirely new mobile structure with the rotated and moved coords 
def align(static, mobile):

    # obtain the NCC coords in COG frame from specified residues 
    # return as a list of numbers x1,y1,z1,x2,y2,z2,... to be aligned

    staticCOG, staticXYZ = static.extractXYZ('post')
    mobileCOG, mobileXYZ = mobile.extractXYZ('pre')


    # check staticXYZ and mobileXYZ are the same length
    if len(staticXYZ) != len(mobileXYZ):
        print( 'Mismatch in number of residues to align between:' )
        print( len(static.postResList) )
        print( len(staticXYZ) )
        print( static.name )
        print( len(mobile.preResList) )
        print( len(mobileXYZ) )
        print( mobile.name )
        sys.exit(0)

    # none of the atoms are permutable
    permlist = []

    # set up minimisation and do the alignment
    mindist = MinPermDistAtomicCluster(niter=1000, permlist=permlist, verbose=True, can_invert=False)
    dist, staticXYZ, mobileXYZ = mindist(staticXYZ, mobileXYZ)

    try:
        rot = mindist.rotbest
    except:
        rot = mindist.mxbest
       
    # obtain the coordinates of all mobile atoms
    allMobileXYZ = mobile.extractAllXYZ()

    # translate and rotate the entire fragment using same rotation matrix.
    mindist.transform.translate(allMobileXYZ, -mobileCOG)
    mindist.transform.rotate(allMobileXYZ, rot)
    mindist.transform.translate(allMobileXYZ, staticCOG)

    # generate a new copy of the input mobile fragment
    newMobile = cp.deepcopy(mobile)

    # update the new mobile with the new XYZ coords.
    newMobile.updateXYZ(allMobileXYZ)

    # return the newMobile to the upper echelons
    return newMobile


def writePDB(fragments, filename):

    # open data file
    try:
        fileHandle = open(filename, 'w')
    except:
        raise Exception("Unable to open output file: " + filename)

    chainNames = [fragment.getChainNames() for fragment in fragments]

    # create a single flattened list of unique chain names
    chainNamesAll=sorted(list(set([ item for sublist in chainNames for item in sublist])))
    
    # initialise counters
    atomNum = 1
    residueNum = 1
    fragmentAtoms = []
    # output each unique chain in the entire construct one chain at a time
    for chain in chainNamesAll:
        for fragment in fragments:
            # write all the information in the current lists of fragment for the given chain, adding caps to that chain if requested 
            # start residue and atomic numbering from atomNum and residueNum.
            # function returns the atomNum and residueNum of the next atom and residue to be written
            [atomNum, residueNum, chainAtoms] = writeChainFragmentToPDB(fileHandle, fragment, chain, atomNum, residueNum)
            fragmentAtoms += chainAtoms
        
        # at the end of each chain write 'TER'
        fileHandle.write('TER\n')

    # at the end of the file write 'END'
    fileHandle.write('END')
    fileHandle.close()


    return fragmentAtoms

def writeAtomGroups(fragments, outfile, atomFile):

    # open data file
    try:
        fileHandle = open(outfile, 'w')
    except:
        raise Exception( "Unable to open output file: " + outfile )

    # open data file
    try:
        fileHandle2 = open(atomFile, 'r')
    except:
        print( '\n\n\n' )
        print( "Unable to open file: " + outfile + "\nSuggest you run tleap on stapled file first and then run this program again." )

        raise Exception( "Program Terminated" )

    atoms = pdb.readAtoms(atomFile)

    # scan through the all fragments and obtain all the distinct chain names throughout the construct
    chainNamesAll = []
    for fragment in fragments:
        # extract all the names of the chains in the current fragment
        chainNames = fragment.getChainNames()
 
        # check each one to see if it's in the big list, if not add it;
        # only add each chain once.
        for chainName in chainNames:
            if not chainName in chainNamesAll:
                chainNamesAll.append(chainName)

    # initialise counters
    fragNumber = 1
    lastResidueInFragment = 0  # simulates the -1th fragment...

    print( 'Exporting AtomGroups:' )

    # Treat each chain independently so atoms groups are only formed by atoms in a single chain.
    # This algorithm assumes the chains are dumped contiguously in the PDB file (which they
    # are in the writePDB function of this program).
    for chain in chainNamesAll:

        print( 'Chain: ' + chain )

        # record the first residue in the entire chain (previous fragment + 1)
        firstResidueInChain = lastResidueInFragment + 1

        # loop through all fragments except the last one
        for fragment in fragments[0:-1]:

            # record the number of the first residue in the Fragment
            firstResidueInFragment = lastResidueInFragment + 1
            # update residueNum to point to last Residue in fragment
            lastResidueInFragment = firstResidueInFragment + fragment.getChainLength(chain) - 1

            print( 'Fragment: ' + str(fragNumber) + ', ' + str(firstResidueInFragment) + ', ' + str(lastResidueInFragment) + ', ' + str(fragment.getChainLength(chain)) )

            # Extract the atom numbers of all atoms in the residues 1st to N-1 residues in chain.
            # Determines bulk of the atoms in the group
            atomsInGroup = []
            for res in range(firstResidueInChain, lastResidueInFragment - 1 + 1):
                [atomsInGroup.append(a) for a in pdb.getAtomsInResidue(atoms, res)]

            # Now find additional atoms, which varies for each group.
            # For each fragment there are four distinct atom groups with well defined rotation axes
            # as per description in Chris forman (wales) book 2, page 1.

            # 1st group:   1st atom in chain to CA of last residue in Frag, does not include R group.
            #              rotation axis: CA and N of last Frag.
            NInLastResidue = pdb.findAtom('N', atoms, lastResidueInFragment)
            HInLastResidue = pdb.findAtom('H', atoms, lastResidueInFragment)
            CAInLastResidue = pdb.findAtom('CA', atoms, lastResidueInFragment)
            GroupAtoms = atomsInGroup
            GroupAtoms.append(NInLastResidue)
            GroupAtoms.append(HInLastResidue)
            GroupAtoms.append(CAInLastResidue)
            axisAtom1 = NInLastResidue
            axisAtom2 = CAInLastResidue
#            print( 'Group One' )
#            print( axisAtom1, axisAtom2 )
#            print( GroupAtoms )
            writeAtomGroup(axisAtom1, axisAtom2, GroupAtoms, fragNumber, 1, fileHandle)

            #  2nd group:   1st atom in chain to N of first residue in next Frag
            #              rotation axis: C of last Frag and N of next Frag.
            #             similar to first group - remove group 1 appended atoms and
            #             add entirety of last residue and N in next residue.
            AllAtomsInLastResidue = pdb.getAtomsInResidue(atoms, lastResidueInFragment)
            NInNextResidue = pdb.findAtom('N', atoms, lastResidueInFragment + 1)
            CInLastResidue = pdb.findAtom('C', atoms, lastResidueInFragment)
            GroupAtoms.remove(NInLastResidue)
            GroupAtoms.remove(HInLastResidue)
            GroupAtoms.remove(CAInLastResidue)
            [GroupAtoms.append(a) for a in AllAtomsInLastResidue]
            GroupAtoms.append(NInNextResidue)
            axisAtom1 = NInNextResidue
            axisAtom2 = CInLastResidue
            writeAtomGroup(axisAtom1, axisAtom2, GroupAtoms, fragNumber, 2, fileHandle)

#           print( 'Group Two' )
#           print( axisAtom1, axisAtom2 )
#           print( GroupAtoms )

            # 3rd group:   1st atom in chain to CA of first residue in next Frag
            #              rotation axis: CA and N of next Frag.
            #              almost same as second group, with the H and CA of next frag.
            HInNextResidue = pdb.findAtom('H', atoms, lastResidueInFragment + 1)
            CAInNextResidue = pdb.findAtom('CA', atoms, lastResidueInFragment + 1)
            GroupAtoms.append(HInNextResidue)
            GroupAtoms.append(CAInNextResidue)
            axisAtom1 = NInNextResidue
            axisAtom2 = CAInNextResidue
            writeAtomGroup(axisAtom1, axisAtom2, GroupAtoms, fragNumber, 3, fileHandle)

#          print( 'Group Three' )
#          print( axisAtom1, axisAtom2 )
#          print( GroupAtoms )
          

            # 4th group:   1st atom in chain to C of first residue in next Frag
            #              rotation axis: CA and C of next frag.
            #             So from group 3 remove CA, N and H of first residue in next frag,
            #             Add entire of first res in next frag, then remove the O.
            #             this adds the R group of the next residue...
            CInNextResidue = pdb.findAtom('C', atoms, lastResidueInFragment + 1)
            OInNextResidue = pdb.findAtom('O', atoms, lastResidueInFragment + 1)
            AllAtomsInNextResidue = pdb.getAtomsInResidue(atoms, lastResidueInFragment + 1)
            AllAtomsInNextResidue.remove(OInNextResidue)
            GroupAtoms.remove(CAInNextResidue)
            GroupAtoms.remove(NInNextResidue)
            GroupAtoms.remove(HInNextResidue)
            [GroupAtoms.append(a) for a in AllAtomsInNextResidue]
            axisAtom1 = CAInNextResidue
            axisAtom2 = CInNextResidue
            writeAtomGroup(axisAtom1, axisAtom2, GroupAtoms, fragNumber, 4, fileHandle)

#            print( 'Group Four' )
#            print( axisAtom1, axisAtom2 )
#            print( GroupAtoms )

            # increment the fragment number
            fragNumber += 1

    # at the end of the file write 'END'
    fileHandle.close()

    return

# write an atomGroups to a file
def writeAtomGroup(axisAtom1, axisAtom2, groupAtoms, frag, group, fH):
    numAtoms = len(groupAtoms) - 2
    firstLine = 'GROUP Frag_' + str(frag) + '_' + str(group) + ' ' + str(axisAtom1) + ' ' + str(axisAtom2) + ' ' + str(numAtoms) + ' 0.1 .3\n'
    fH.write(firstLine)
    for atom in groupAtoms:
        if not atom in [axisAtom1, axisAtom2]:
            fH.write(str(atom) + '\n')
    fH.write('\n')
    return

# Looks up all the bits of a fragment which belong to a specific chain 
# and outputs to a PDB, capping the top or bottom if instructed to do so
# build up an atom list to return as we go
def writeChainFragmentToPDB(fH, frag, chainName, atomNumOut, resNumOut):
    
    # specify the output chain name. THis is normally just chainName unless chainName is _ in which case output an A.
    # total hack for a situation where a pdb has no chain names specified. Just put a _ in the definition file for
    # chain name. Saves having to put an A into a PDB column for chain name. (For spidroin this file was 33000 records.)
    # could have done it vi. i know. but I was on a pc. Could have done it with WSL true. But whatever.  
    outChainName = chainName
    if chainName=='_':
        outChainName='A'

    # specify the output chain name. THis is normally just chainName unless chainName is _ in which case output an A.
    # total hack for a situation where a pdb has no chain names specified. Just put a _ in the definition file for
    # chain name. Saves having to put an A into a PDB column for chain name. (For spidroin this file was 33000 records.)
    # could have done it vi. i know. but I was on a pc. Could have done it with WSL true. But whatever.  
    outChainName = chainName
    if chainName=='_':
        outChainName='A'

    # extract the atoms from the fragment belonging to a particular chain 
    atoms = frag.extractChainAtoms(chainName)
    # array to output list of atoms
    chainAtoms = []
    #check that the current fragment has atoms belonging to the current chain. If not then bug out.
    if atoms:
        # Add an ACE cap if required. estimate the coords for the C of the ACE
        if frag.checkNCap(chainName):
            CPos=frag.estimateNextResidueC(chainName)
            l = 'ATOM {: >06d} {: <4}{:1}{:3} {:1}{: >4d}{:1}   {: >8.3f}{: >8.3f}{: >8.3f}\n'.format(atomNumOut, 'C', '', 'ACE', chainName, int(resNumOut),'',CPos[0],CPos[1],CPos[2])
            fH.write(l)
            chainAtoms.append([atomNumOut, 'C', '', 'ACE', outChainName, int(resNumOut), '', CPos[0], CPos[1], CPos[2], 0.0, 0.0, '', 'C', ''])
            atomNumOut += 1
            resNumOut += 1
     
        # get the first residue number from the fragment
        fragResNumPrev = atoms[0][5]
    
        # loop through each atom
        for atom in atoms:
      
            # set the atom number in the current output record
            atom[0] = atomNumOut
    
            # make a note of the number of the current residue in the original data
            fragResNumCur = atom[5]
    
            # only increment the resNumberOut when the residue number increments in the original data (doesn't increment last residue)
            if fragResNumCur != fragResNumPrev:
                resNumOut = resNumOut + 1
                fragResNumPrev = fragResNumCur
    
            # set the residue number
            atom[5] = resNumOut
    
            # set the chain
            atom[4] = outChainName
    
            # eliminate an annoying carriage return
            atom[14] = ' '
     
            # write the atom to the file
            l = pdb.pdbLineFromAtom(atom)  # includes carriage return
            chainAtoms.append(atom)
            fH.write(l)
            atomNumOut += 1
    
        # check to see if at least one atom was written to file. If So increment the residueNumber by one
        # ready for next call
        if len(atoms) > 1:
            # increment residue number 
            resNumOut += 1
     
        # output a CCAp if required
        if frag.checkCCap(chainName):
            NPos=frag.estimateNextResidueN(chainName)
            l = 'ATOM {: >06d} {: <4}{:1}{:3} {:1}{: >4d}{:1}   {: >8.3f}{: >8.3f}{: >8.3f}\n'.format(atomNumOut, 'N', '', 'NME', chainName, int(resNumOut),'',NPos[0],NPos[1],NPos[2])
            fH.write(l)
            chainAtoms.append([atomNumOut, 'N', '', 'NME', outChainName, int(resNumOut), '', NPos[0], NPos[1], NPos[2], 0.0, 0.0, '', 'N', ''])
            atomNumOut += 1
            resNumOut += 1

    return atomNumOut, resNumOut, chainAtoms

def exploreAngles():
    try:
        os.chdir('angles')
    except OSError:
        os.system("mkdir angles")

    os.system("mv /home/cjf41/HETS/H12/atomgroups /home/cjf41/HETS/H12/angles/.")
    os.system("mv /home/cjf41/HETS/H12/coords.inpcrd /home/cjf41/HETS/H12/angles/.")
    os.system("mv /home/cjf41/HETS/H12/coords.prmtop /home/cjf41/HETS/H12/angles/.")
    os.system("cp /home/cjf41/lib/data/min_md.in /home/cjf41/HETS/H12/angles/.")
    os.system("cp /home/cjf41/lib/data/min.in /home/cjf41/HETS/H12/angles/.")
    os.system("cp /home/cjf41/lib/data/dataQuenchStaple /home/cjf41/HETS/H12/angles/data")
    os.system("cp /home/cjf41/lib/data/pbsscriptQuenchStaple /home/cjf41/HETS/H12/angles/pbsscript")

    # should be everything! move into directory and submit the script
    os.chdir('/home/cjf41/HETS/H12/angles')

#    os.system("qsub pbsscript")
    os.system("AMBGMIN &")
    os.chdir('/home/cjf41/HETS/H12/')

    return

def doRotations(rotations, atoms):

    # create lists of residue nums about whose C and CA we must rotate, and the angle to rotate by 
    residueNums = [ int(r[0]) for r in rotations]
    angles = [ float(r[1])*np.pi/180 for r in rotations]
    
    # extract a numpy array of the atom positions
    atomXYZ = np.array([[atom[7], atom[8], atom[9]] for atom in atoms ])
        
    # get a list of the residNums and atom type for each atom.
    resAtoms = [[int(atom[5]), atom[1]] for atom in atoms]

    # get the index of the Oxygen in each residue of interest.    
    OIndices = [ resAtoms.index([res,'O']) for res in residueNums ]
    
    # get the index of the CA in each residue of interest.
    CAIndices = [ resAtoms.index([res,'CA']) for res in residueNums ]
    
    # get the index of the C in each residue of interest.
    CIndices = [ resAtoms.index([res,'C']) for res in residueNums ]
    
    print(len(CIndices), len(CAIndices), len(OIndices), len(angles), len(residueNums) )        

    # loop through all the specified rotations
    for CIndex, CAIndex, OIndex, angle, resNum  in zip(CIndices, CAIndices, OIndices, angles, residueNums):
        print( "Processing List from resnum: ", resNum, angle, OIndex, CIndex, CAIndex )
        
        # Compute the NVec, from the latest array of positions
        nVec = atomXYZ[CIndex] -atomXYZ[CAIndex]
        nVec = nVec/np.linalg.norm(nVec)

        # update the coords that need replacing
        atomXYZ[OIndex:] = [ rotPAboutAxisAtPoint( p, atomXYZ[CIndex], nVec, angle) for p in atomXYZ[OIndex:] ]

    print( "updating atom array" )
    for i, pos in enumerate(atomXYZ):
        atoms[i][7]= pos[0]
        atoms[i][8]= pos[1]
        atoms[i][9]= pos[2] 

    # return the new atoms array
    return atoms


# Main Routine
if __name__ == '__main__':
    """ STAPLER

    Complile building block fragments of PDB into a single larger protein using an instruction file containing a 
    list of fragments and staples as appropriate. A root building block starts the process. A staple is a fragment of 
    protein structure that is used to align subsequent building blocks to the growing structure.
    
    One part of the staple fragment is aligned to the root building block.  Then the next building block is aligned to 
    a subset of the staple. The staple is discarded and the two aligned building blocks are saved as single PDB.
    
    The system can cope with multiple chains but all staples and building blocks must have the same number of chains.  
    E.g it was used to build up triple helix collagen building blocks once upon a time.     
    
    Command line syntax:      

    pdbStapler.py <instructionFile>

    example instruction file: 
    
    OUTPUTFILE H12.pdb

    FRAGMENT GS.pdb
    CHAIN A 1 2
    CHAIN B 3 4
    PREV Terminal
    NEXT 2 4
    NCAP None
    CCAP None
    END

    STAPLE gsl1.pdb
    CHAIN A 1 2
    CHAIN B 3 4
    PREV 1 3
    NEXT 2 
    END

    FRAGMENT L1.pdb
    CHAIN A 218 226
    PREV 218
    NEXT 226
    NCAP None
    CCAP None
    END
    
    ROTATION 224 180
    
    
    The first line of the instruction is the output filename.
    
    A FRAGMENT keyword indicates the pdb file that defines a building block. Subsequent lines specifies the parameters necessary to connect the building block.
    Each block must have a terminating END line.   
    
    All the atoms from the input PDB are read in verbatim and output into the final structure with new coords and index numbers. 
    Each residue in the input file must have a unique residue number.
     
    CHAIN specifies the start and end residues of each chain in the fragment as identified using the native numbering 
    system in the pdb file. This information is used to label chains correctly in the final output PDB. THe actual rotation alignment 
    process ignores chains and simply aligns a specified set of residues with the next building block/staple in the procedure. If there is no chain name 
    specified in the pdb use an underscore as the chain name: _ 
    
    As many residues as we like can be used in the alignment at each step, but the same number of residues must be specified in the connecting
    sub sets of each building block/staple.  e.G. i can read in a 100 residue protein which might have multiple chains. All residues have their own ref number.  
    I can specify any of the residues in the entire process to use in an alignment step.  I must specifiy the same number of residues in the next building block/staple
    to align. There is no reason why each alignment set needs the same number of residues in every part of the process.     
    
    PREV specifies a set of residues from this fragment which are to be aligned with a set of the same size in the previous object.
    NEXT specifies a set of residues from this fragment which are to to align with a set of the same size in the next object. 
    
    NCAP specifies the names of the chains which will have an N terminal acetyl cap. 
    CCAP specifes the chains which will have an C terminal cap.
    
    For the caps only a place holder N atom or C atom is inserted without specifying atomic co-ordinates. The structure should be run through Amber tools tleap to build the 
    terminal groups in a sensible way.  A set of after effects can be added which includes a PAG flag which will run the structure through tleap if it is installed on the 
    system.

    The routine then recurses through the list. First fragment is kept fixed. the "prev" residues of the staple are aligned
    with the set of next residues of the stationary fragment. These sets of atoms must be the same size.  
    Same size means same num of residues, only N CA and C are aligned. 
    
    Then the staple in its new position is kept fixed, and the next fragment
    is aligned with the staple. The prev residues of the fragment are aligned with the next residues of the staple. 

    The alignment takes the N CA and C atoms of all the specified residues and aligns all of them  
    at the same time.

    The rotation matrix is then acquired and used to rotate all the atoms in the Fragment simultaneously.

    It is up to the user to ensure the adjacent fragments/staples are consistent with each other.

    All the stuff with chains is about making disparate chains in distinct fragments line up, but then be  
    labelled in the output as continuous chains with the same letter. This was modification for collagen.
    
    Eg a block with three chains, can be aligned with another block containing three chains, and the output will
    consist of three chains, each labelled separately.  

    A ROTATION command species a residue in the fully stapled structure which identifies an axis between 
    a specific C and CA bond and rotates all atoms from the C onwards to the end of the fully stapled chain
    by that amount.  One can specify as many rotations as you like. This is useful to help blocks from 
    overlapping.   
    
    No checking of the sanity of the final structure is employed. IT's up to the user to ensure that sensible results occur.
    
    A set of after effects can be specified including running the structure through tleap.
    
    These are included as command line parameters after the main parameter file
    
    PAG
    relax
    atomgroups AGfilename
    
    PAG specifies that the Prepare for Amber GMin routine from PDB proc will be run on the stapler output. This generates a coords file and parameter file 
    as well as stripped out hydrogens and populating the structure with missing heavy atoms.  

    relax caused a bunch of steps to be taken with AMBGMIN. THis is hard coded for one thing I wanted to do one day. It could be generalised from this 
    prescription. 

    If an atomgroups file is request then an atomgroups file is generated using the instructions contains in the file AFGilename. This specifies regions of 
    the final structure that can be thought of rigid building blocks for rotations.  In this way we can use building blocks and 
    explore the structures in GMIN and OPTIM etc using the rigid bodies framework. I.e. Treat each building block input into stapler as a coarse grain, 
    even though they are atomistic and rotate it as specified. Relax uses the GMIN system to explore the building block rotations to help find sensible structures.
    """


    # load the instruction file which contains two lists 
    # a list of fragments and a list of staples In ORDER. 
    print( "loading data" )
    [outFile, fragments, staples, rotations, afterEffects] = getInput()


    print( "Stapling the following fragments:" )
    for fragment in fragments:
        print( fragment.name , "\n")

    print( "\nusing the following staples:" )
    for staple in staples:
        print( staple.name , "\n")

    print( '\nOutput filename:' )
    print( outFile )

    print( '\nRotations:' )
    print( rotations )


    # count the staples
    numStaples = len(staples)

    # initialise output list (alignedFragments) with a copy of the first fragment
    alignedFragments = [fragments[0]]
    alignedStaples = []

    # for the remaining fragments, first align the staple, then align the new fragment
    for curItem in range(0, numStaples):
        # the function align keeps the object referred to in the first argument static 
        # and rotates the second one. A entirely new object is returned which is 
        # identical in all respects except with new translated and rotated atomic coordinates.

        print("Perform stapling operation", curItem + 1, " of  ", numStaples)
    
        print("Performing Pre-Operation")

        # align staple to fragment
        alignedStaple = align(alignedFragments[curItem], staples[curItem])
        alignedStaples.append(alignedStaple)

        print("Performing Post-Operation")
        # using the previously rotated staple as a static construct, now align next fragment to that. Add aligned fragment to the list
        alignedFragment = align(alignedStaple, fragments[curItem + 1])
        alignedFragments.append(alignedFragment)

    # output the pdb files
    writePDB(alignedStaples, 'staples.pdb')
    atoms = writePDB(alignedFragments, outFile)
    
    # loads the combined pdb and performs rotations as defined. Totally ignores chains.
    # just goes by residue number. Worked for what I wanted at the time. Soz. 
    atoms = doRotations(rotations, atoms)
    
    # convert atoms chain to strings
    l_atoms = [pdb.pdbLineFromAtom(atom) for atom in atoms]
    
    # write strings to file with a rot_<outfile> suffix
    pdb.writeTextFile( l_atoms, outFile[0:-4] + '_rot.pdb')

    # control behaviour from command line
    PAG = afterEffects[0]
    atomgroups = afterEffects[1]
    AGFilename = afterEffects[2]
    relax = afterEffects[3]

    if PAG == 1:
        # run the pdb through tleap
        pdb.prepAmberGMin(outFile, ['/home/cjf41/lib/data/rulefile', 'ff99SB', '/home/cjf41/lib/data/prepHYP05.in', '/home/cjf41/lib/data/HYP05.forcemod'])
 
    if atomgroups == 1:
        # output an atom groups file which treats each fragment as a 
        # rigid body with four degrees of freedom per fragment as defined 
        # in Page 1 of book 2, Chris Forman (wales) lab notes.
        writeAtomGroups(alignedFragments, 'atomgroups', AGFilename)

    if relax == 1:
        # explore all the angles and see if we can't fold the protein sensibly
        exploreAngles()

    print( "Done" )

