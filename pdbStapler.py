#!/usr/bin/env python
import sys, os
sys.path.append('/usr/lib64/python2.7/site-packages')
sys.path.append('~/src/pele')
sys.path.append('/home/cjf41/HETS/H12/')
sys.path.append('/home/cjf41/lib/data')

import math as math
import numpy as np
import pdbLib as pdb
import copy as cp
from pele.mindist import MinPermDistAtomicCluster
from pele.utils.rotations import random_q
from pele.utils.rotations import q2mx
from pele.utils.rotations import mx2aa
from pele.utils.rotations import aa2mx



class chain:
    """a class for storing information about a chain"""
    def __init__(self, chainName, startResidue, endResidue):
        self.chainName = chainName
        self.startResidue = startResidue
        self.endResidue = endResidue
        self.chainLength = endResidue - startResidue + 1

    def Display(self):
        print self.chainName
        print self.startResidue
        print self.endResidue
        print self.chainLength
        return

    # assumes residue numbers in a chain are monotonic and contiguous
    def checkResidueInChain(self, resNum):
        return (resNum >= self.startResidue) and (resNum <= self.endResidue)

    def getChainName(self):
        return self.chainName
 
    def getChainLength(self):
        return self.chainLength

class fragment:
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
        print "Chain Names:"
        print self.chainNames
        print "Chain Info:"       
        for chain in self.chains:
            chain.Display()
        print "Filename:"
        print self.name
        print "Number of atoms"
        print len(self.atoms)
        print "Residues Attaching to Previous Fragment"
        print self.preResList
        print "Residues Attaching to Next Fragment"
        print self.postResList
        print "NCap chains"
        print self.NCapList
        print "CCap chains"
        print self.CCapList
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
        return [atom for atom in self.atoms if atom[4]==chainName]

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
            print 'Unable to update atomic XYZ; inconsistent number of atoms'
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
        raise Exception, "Unable to open input file: " + filename
        sys.exit(0)

    # read in the text defined in pdbLib.py
    instructions = pdb.readTextFile(filename)

    # parse the input instructions
    outfile, fragments, staples = parseInstructions(instructions)

    proceed, errorMsg = validateInputConsistency(fragments, staples)
    
    print fragments
    print staples
    

    if proceed == 1:
        print 'Unable to continue; input inconsistent:'
        print errorMsg
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

    return outfile, fragments, staples, [PAG, atomgroups, AGFilename, relax]

# function to parse an instructions file read in as text
def parseInstructions(instructions):

    # initialise output variables
    fragments = []
    staples = []
    outfile = []

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
                fragments.append(fragment(chains, preResList, postResList, NCap, CCap, infile))
            else:
                staples.append(fragment(chains, preResList, postResList, NCap, CCap, infile))
    
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
                
    return outfile, fragments, staples

def validateInputConsistency(fragments, staples):

    numFrags = len(fragments)
    numStaples = len(staples)

    errorMsg = ''

    # initialise the proceed variable. Zero good. Assume success unless fail.
    proceed = 0
    if numFrags - numStaples != 1:
        proceed = 1
        errorMsg = 'Wrong number of staples or fragments. nF=nS+1.'

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
        print 'Mismatch in number of residues to align between:'
        print len(mobileXYZ)
        print mobile.name
        print len(staticXYZ)
        print static.name
        sys.exit(0)

    # none of the atoms are permutable
    permlist = []

    # set up minimisation and do the alignment
    mindist = MinPermDistAtomicCluster(niter=1000, permlist=permlist, verbose=True, can_invert=False)
    dist, staticXYZ, mobileXYZ = mindist(staticXYZ, mobileXYZ)

    try:
        rot = mindist.rotbest
    except:
        rot=mindist.mxbest
       
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
        raise Exception, "Unable to open output file: " + filename

    # scan through the all fragments and obtain all the distinct chain names throughout the construct, uses the set functionallity
    chainNamesAll=sorted(list(set([ item for sublist in [fragment.getChainNames() for fragment in fragments] for item in sublist])))
    
    # initialise counters
    atomNum = 1
    residueNum = 1

    # output each unique chain in the entire construct one chain at a time
    for chain in chainNamesAll:
        for fragment in fragments:
            # write all the information in the current lists of fragment for the given chain, adding caps to that chain if requested 
            # start residue and atomic numbering from atomNum and residueNum.
            # function returns the atomNum and residueNum of the next atom and residue to be written
            [atomNum, residueNum] = writeChainFragmentToPDB(fileHandle, fragment, chain, atomNum, residueNum)

        # at the end of each chain write 'TER'
        fileHandle.write('TER\n')

    # at the end of the file write 'END'
    fileHandle.write('END')
    fileHandle.close()


    return

def writeAtomGroups(fragments, outfile, atomFile):

    # open data file
    try:
        fileHandle = open(outfile, 'w')
    except:
        raise Exception, "Unable to open output file: " + outfile

    # open data file
    try:
        fileHandle2 = open(atomFile, 'r')
    except:
        print '\n\n\n'
        print "Unable to open file: " + outfile + "\nSuggest you run tleap on stapled file first and then run this program again."

        raise Exception, "Program Terminated"

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

    print 'Exporting AtomGroups:'

    # Treat each chain independently so atoms groups are only formed by atoms in a single chain.
    # This algorithm assumes the chains are dumped contiguously in the PDB file (which they
    # are in the writePDB function of this program).
    for chain in chainNamesAll:

        print 'Chain: ' + chain

        # record the first residue in the entire chain (previous fragment + 1)
        firstResidueInChain = lastResidueInFragment + 1

        # loop through all fragments except the last one
        for fragment in fragments[0:-1]:

            # record the number of the first residue in the Fragment
            firstResidueInFragment = lastResidueInFragment + 1
            # update residueNum to point to last Residue in fragment
            lastResidueInFragment = firstResidueInFragment + fragment.getChainLength(chain) - 1

            print 'Fragment: ' + str(fragNumber) + ', ' + str(firstResidueInFragment) + ', ' + str(lastResidueInFragment) + ', ' + str(fragment.getChainLength(chain))

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
#            print 'Group One'
#            print axisAtom1, axisAtom2
#            print GroupAtoms
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

#           print 'Group Two'
#           print axisAtom1, axisAtom2
#           print GroupAtoms

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

#          print 'Group Three'
#          print axisAtom1, axisAtom2
#          print GroupAtoms
          

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

#            print 'Group Four'
#            print axisAtom1, axisAtom2
#            print GroupAtoms

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
def writeChainFragmentToPDB(fH, frag, chainName, atomNumOut, resNumOut):

    # extract the atoms from the fragment belonging to a particular chain 
    atoms = frag.extractChainAtoms(chainName)

    #check that the current fragment has atoms belonging to the current chain. If not then bug out.
    if atoms:
        # Add an ACE cap if required. estimate the coords for the C of the ACE
        if frag.checkNCap(chainName):
            CPos=frag.estimateNextResidueC(chainName)
            l = 'ATOM {: >06d} {: <4}{:1}{:3} {:1}{: >4d}{:1}   {: >8.3f}{: >8.3f}{: >8.3f}\n'.format(atomNumOut, 'C', '', 'ACE', chainName, int(resNumOut),'',CPos[0],CPos[1],CPos[2])
            fH.write(l)
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
            atom[4] = chainName
    
            # eliminate an annoying carriage return
            atom[14] = ' '
     
            # write the atom to the file
            l = pdb.pdbLineFromAtom(atom)  # includes carriage return
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
            atomNumOut += 1
            resNumOut += 1

    return atomNumOut, resNumOut

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



# Main Routine
if __name__ == '__main__':
    """ STAPLER

    Complile fragments of PDB into a single protein using an instruction file containing a 
    list of fragments and staples as appropriate.

    pdbStapler.py <instructionFile>

    example instruction file: 
    first line is the output filename.
    Then we have a fragment block which has the data filename for that block. 
    The start and end residues of each chain are identified using the numbering 
    system in the pdb file.
    PREV specifies the list of residues (one for each chain) which links to the previous molecule.
    NEXT specifies the list of residues (one for each chain) which links to the next molecule.
    NCAP specifies the chains which will have an N terminal acetyl cap. 
    CCAP specifes the chains which will have an C terminal cap.
    for the caps only a place holder N atom or C atom is inserted without specifying atomic co-ordinates. tleap can
    add the remaining cap atoms in a sensible way.

    routine then recurses through the list. first fragment is kept fixed. the "prev" residues of the staple are aligned
    with the next residue of the stationary fragment.   Then the staple in its new position is kept fixed, and the next fragment
    is aligned with the staple. The prev residues of the fragment are aligned with the next residues of the staple. 

    The alignment takes the N CA and C atoms of all the residues (one from each chain) and aligns all the residues at the same time.

    The rotation matrix is then acquired and used to rotate all the atoms in the residue simultaneously.

    It is up to the user to ensure the adjacent fragments/staples are consistent with each other.

 
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
    END"""


    # load the instruction file which contains two lists 
    # a list of fragments and a list of staples In ORDER. 
    [outFile, fragments, staples, afterEffects] = getInput()


    print "Stapling the following fragments:"
    for fragment in fragments:
        print fragment.name

    print "\nusing the following staples:"
    for staple in staples:
        print staple.name

    print '\nOutput filename:'
    print outFile

    # count the staples
    numStaples = len(staples)

    # initialise output list with a copy of the first fragment
    alignedFragments = [fragments[0]]
    alignedStaples = []

    # for the remaining fragments, first align the staple, then align the new fragment
    for curItem in range(0, numStaples):
        # the function align keeps the object referred to in the first argument static 
        # and rotates the second one. A entirely new object is returned which is 
        # identical in all respects except with new translated and rotated atomic coordinates.

        # align staple to fragment
        alignedStaple = align(alignedFragments[curItem], staples[curItem])
        alignedStaples.append(alignedStaple)

        # using the previously rotated staple as a static construct, now align next fragment to that. Add aligned fragment to the list
        alignedFragment = align(alignedStaple, fragments[curItem + 1])
        alignedFragments.append(alignedFragment)

    # output the pdb files
    writePDB(alignedStaples, 'staples.pdb')
    writePDB(alignedFragments, outFile)


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

    print "Done"




