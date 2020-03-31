#! /usr/bin/env python

import sys
sys.path.append('/usr/local/lib64/python2.4/site-packages')

#from math import *
from numpy import *
import scipy
scipy.pkgload('optimize')
import os
import subprocess
import shutil
import time
import random
import copy

def getFilename():
    if len(sys.argv) != 2:  # the program name and the one argument
      # stop the program and print an error message
      sys.exit("analysePDB XXXX  This is the PDB 4 digit code. Must also create the file XXXX.sit expressing residues of interest on seperate lines, and also the file XXXX.pair indicating which pair of building blocks is the primary pair.")
    return sys.argv[1]

# can input p as an array or as a matrix; function always returns a matrix
def aaxis_to_mat(p):
    p=matrix(p)
    """Converts an angle-axis rotation vector into a rotation matrix"""
    # cf wikipedia page for Rodrigues's rotation formula
    theta = linalg.norm(p)
    if abs(theta)< 1e-9:
        return matrix(identity(3))
    else:
        k = p / theta
        kx = matrix([[0, -k[0,2], k[0,1]], [k[0,2], 0, -k[0,0]], [-k[0,1], k[0,0], 0]])
        return identity(3) * cos(theta) + kx * sin(theta) + (1 - cos(theta)) * outer(k, k)

#input rm in a matrix or array form; returns p as an array
def mat_to_aaxis(rm):
    """Converts a rotation matrix into an angle axis"""
    #convert to matrix form
    rm=matrix(rm)
    #formula 
    p=array([rm[2,1]-rm[1,2], rm[0,2]-rm[2,0], rm[1,0]-rm[0,1]])
    r=linalg.norm(p)
    
    try:
       theta=math.acos(0.5*(rm[0,0]+rm[1,1]+rm[2,2] - 1.0 ))
    except ValueError:
       #print 'argument magnitude exceed limit by: ',-1.0-0.5*(rm[0,0]+rm[1,1]+rm[2,2] - 1.0 )
       theta=math.acos(-1.0)

    #check for conditions where it's ok to divide by r
    if (r>1e-9):
        #normalise P and scale by theta
        p=theta*p/r    
    else:
        #check for identity matrix
        if (abs(rm[0,0]+rm[1,1]+rm[2,2]-3)<1e-9)and(abs(rm[0,1]+rm[1,0])<1e-9)and(abs(rm[0,2]+rm[2,0])<1e-9)and(abs(rm[1,2]+rm[2,1])<1e-9):
            p=array([0,0,2*pi])
        else:
            #singular matrix but not identity so 180 degree rotation about some axis
            #Do some checks and figure it out (pretty unlikely) - borrowed from some webpage
            xx=(rm[0,0]+1)/2
            yy=(rm[1,1]+1)/2
            zz=(rm[2,2]+1)/2
            xy=(rm[0,1]+rm[1,0])/4
            xz=(rm[0,2]+rm[2,0])/4
            yz=(rm[1,2]+rm[2,1])/4
            if (xx>yy)and(xx>zz):
               if xx<1e-9:
                   x=0
                   y=1/sqrt(2)
                   z=1/sqrt(2)
               else:
                   x=sqrt(xx)
                   y=xy/x
                   z=xz/x
            elif (yy>zz):
               if yy<1e-9:
                   x=1/sqrt(2)
                   y=0
                   z=1/sqrt(2)
               else:
                   y=sqrt(yy)
                   x=xy/y
                   z=yz/y
            else:
                if zz<1e-9:
                   x=1/sqrt(2)
                   y=1/sqrt(2)
                   z=0
                else:
                   z=sqrt(zz)
                   x=xz/z
                   y=yz/z
            p=array([x,y,z])
            r=linalg.norm(p)
            p=pi*p/r
    return p

#filename is a string, data is in this format:  array([ array{[,,], matrix([[,,],[,,],[,,]]) ])
#orientation is a flag, true or false. if truye outputs orientations, if false only outputs positions.
def writeCoords(filename, data, orientations):
    #data format:  [  [pos1,rmat1], [pos2,rmat2], [pos3,rmat3]....]
    f=open(filename,'w')
    coords=[ str(pos[0][0])+' '+str(pos[0][1])+' '+str(pos[0][2])+'\n' for pos in data]
    for pos in coords:
       f.write(pos)
    #if required output orientations
    if (orientations):
        pList=[mat_to_aaxis(rotmat[1]) for rotmat in data]
        pString=[str(p[0])+' '+str(p[1])+' '+str(p[2])+'\n' for p in pList]
        for p in pString:
            f.write(p)
    f.close()
    return


#data output in format:  [  [pos1,rmat1], [pos2,rmat2], [pos3,rmat3]....] 
#where posi is an array([]) and rmati is a matrix([[],[],[]])
def readCoords(filename):
    """Parses coords assuming just xyz"""
    f = open(filename, 'r')
    coords=[array([float(x) for x in line.split()]) for line in f.readlines()]
    numCoords=len(coords)/2
    xyz=coords[0:numCoords]
    pVecs=coords[numCoords:2*numCoords]
    return [[pos, aaxis_to_mat(p)] for (pos,p) in zip(xyz,pVecs)]



#data is input as a list of building blocks [bb1,  bb2, ... ] 
#where a building block bbi is a list of sites [site1, site2, ...]
#where sitei is a list of ellipsoids [ellipsoid1,ellipsoid2,...]
#where ellipsoidi is a list of attributes [ pos, size, rot ]
#where pos is array([x,y,z])
#size is array([ra,rb,rc,aa,ab,ac])
#and rmat is matrix([[],[],[]])
def writeXYZ(filename,data,energy):
    f=open(filename,'w')
    numBBs=len(data)
    numSites=len(data[0])
    numEPerSite=len(data[0][0])
    numAtomsStr=str(numBBs*numSites*numEPerSite)+'\n'
    f.write(numAtomsStr)
    f.write('Energy of minimum      1= '+str(energy)+' first found at step        0\n')
    atom=['O','N','C']
    overRide=0
    for bb in data:
       for site in bb:
          atomIndex=0
          for ellipsoid in site:
            x=str(ellipsoid[0][0])
            y=str(ellipsoid[0][1])
            z=str(ellipsoid[0][2])
            ar=str(ellipsoid[1][0])
            br=str(ellipsoid[1][1])
            cr=str(ellipsoid[1][2])
            a11=str(ellipsoid[2][0,0])
            a12=str(ellipsoid[2][0,1])
            a13=str(ellipsoid[2][0,2])
            a21=str(ellipsoid[2][1,0])
            a22=str(ellipsoid[2][1,1])
            a23=str(ellipsoid[2][1,2])
            a31=str(ellipsoid[2][2,0])
            a32=str(ellipsoid[2][2,1])
            a33=str(ellipsoid[2][2,2])
            p=mat_to_aaxis(ellipsoid[2])
            p1=str(p[0])
            p2=str(p[1])
            p3=str(p[2])
            if overRide>0:
                atomIndex=overRide
            atomStr=atom[atomIndex]
            vectString=atomStr+' '+x+' '+y+' '+z+' ellipse '+ar+' '+br+' '+cr+' '+a11+' '+a12+' '+a13+' '+a21+' '+a22+' '+a23+' '+a31+' '+a32+' '+a33+' atom_vector '+p1+' '+p2+' '+p3+'\n'
            atomIndex+=1
            if atomIndex==3:
                atomIndex=1
            f.write(vectString)
    f.close()
    return

#Finds the index of every MODEL and ENDMDL line in the PDB file
#Reads in all the atoms lines between those indices
#breaks each line up in to individual substrings
def readPDB(filename):
    f=open(filename,'r')
    inp=f.readlines()
    inpSplit=[x.split() for x in inp]
    sIndex=[inpSplit.index(x) for x in inpSplit if x[0] in ['MODEL']]
    eIndex=inpSplit.index(['ENDMDL'])
    lModel=eIndex-sIndex[0]-1
    atoms=[inpSplit[si+1:si+lModel+1] for si in sIndex]
    f.close()
    return atoms[0]

#write a building block out into into a pysites file.
#The values for the ellipsoid sizes in a standard xyz file are semi axes.
#This program uses full axes throughout, so divide by two here.
#format of a buiilding block is a list of sites [site1, site2, ...]
#where sitei is a list of ellipsoids [ellipsoid1,ellipsoid2,...]
#where ellipsoidi is a list of attributes [ pos, size, rot ]
#where pos is array([x,y,z])
#size is array([ra,rb,rc,aa,ab,ac])
#and rmat is matrix([[],[],[]])
#matrix is converted and output as an angle axis
def writePYSites(filename,BB,rOn,aOn):
    f=open(filename,'w')
    numSitesStr=str(len(BB))+'\n'
    f.write(numSitesStr)
    f.write('analysePDB\n')
    for site in BB:
      for ellipsoid in site:
         x=str(ellipsoid[0][0])
         y=str(ellipsoid[0][1])
         z=str(ellipsoid[0][2])
         rep=str(rOn)
         att=str(aOn)
         ar=str(ellipsoid[1][0]/2.0)
         br=str(ellipsoid[1][1]/2.0)
         cr=str(ellipsoid[1][2]/2.0)
         aa=str(ellipsoid[1][3]/2.0)
         ba=str(ellipsoid[1][4]/2.0)
         ca=str(ellipsoid[1][5]/2.0)
         p=mat_to_aaxis(ellipsoid[2])
         p1=str(p[0])
         p2=str(p[1])
         p3=str(p[2])
         vectString='O '+x+' '+y+' '+z+' ellipse '+rep+' '+att+' '+ar+' '+br+' '+cr+' '+aa+' '+ba+' '+ca+' atom_vector '+p1+' '+p2+' '+p3+'\n'
         f.write(vectString)
    f.close()
    return

#reads in a list of residues within each protein monomer that belong
#to a particular site. The residues are organised on aline by line basis
#in the file. thus each site is given a unique entry in the output list.
def readSites(filename):
    f=open(filename,'r')
    inp=f.readlines()
    f.close()
    return [x.split() for x in inp]

#reshape the list of atoms from the PDB so each monomer/strand has a unique entry
#eventually each monomer/strand will have it's own building block representation.
def findStrands(atoms):
    #identify unique strands
    seq=sorted(list(set([x[4] for x in atoms if x[0] not in ['TER']])))
    #group the strands
    return list([[x for x in atoms if x[4]==u] for u in seq])

#analyse the data from the protein data base.
#The structure is broken down into building blocks.
#Each building block is broken down into distinct beta strand 
#sites which can be represented by an ellipsoid.
#The residues contributing to each ellipsoid are decided on in advance by a human.
#The parameters of each ellipsoid are estimated and stored in a list.
#Returns a list of ellipsoids organised by site and building block.
#Format of output list is a list of building blocks [bb1,  bb2, ... ] 
#where a building block bbi is a list of sites [site1, site2, ...]
#where sitei is a list of ellipsoids [ellipsoid1,ellipsoid2,...]
#where ellipsoidi is a list of attributes [ pos, size, rot ]
#where pos is array([x,y,z])
#size is array([ra,rb,rc,aa,ab,ac])
#and rmat is matrix([[],[],[]])
def analyseXYZ(siteResI,allAtoms):
    #reorganise list of atoms in the current model by strand
    strands=findStrands(allAtoms)
    #select from each strand the positions of the atoms which belong to each site
    xyz=[[[array([float(atom[6]),float(atom[7]),float(atom[8])]) for atom in strand if atom[5] in curSite] for curSite in siteResI] for strand in strands]
    #sum average position of all atoms in each site (COG)
    COG=[[sum(curSite,axis=0)/len(curSite) for curSite in strand] for strand in xyz]
    #find displacement of each atom from the centre of gravity of each site
    dR=[[[atom-cSite for atom in aSite] for (aSite,cSite) in zip(aStrand,cStrand)] for (aStrand,cStrand) in zip(xyz,COG)]
    #compute contribution of each particle to the (geometric) moments of inertia of each site.
    I=[[[ matrix([[dRi[1]**2+dRi[2]**2, -dRi[0]*dRi[1], -dRi[0]*dRi[2]],[ -dRi[0]*dRi[1], dRi[0]**2+dRi[2]**2,-dRi[1]*dRi[2]],[-dRi[0]*dRi[2],-dRi[1]*dRi[2], dRi[0]**2+dRi[1]**2]]) for dRi in curSite] for curSite in strand] for strand in dR]
    iSum= [[sum(curSite,0) for curSite in strand] for strand in I]
    #diagonalise moments of inertia to get principle axes and principle moments of inertia 
    #for each ellipsoid. The eigenvectors are written in the same coordinate system 
    #as the protein database co-ordinates, which is taken as the lab frame. 
    eInfo=[[linalg.eig(matrix(curI)) for curI in strand] for strand in iSum]

   
    #Project all the atoms in each site onto the resulting axes
    siteXYZ=[[[ array((curAtom*E[1]).tolist()[0]) for curAtom in aSite] for (aSite,E) in zip(aStrand,eStrand)] for (aStrand,eStrand) in zip(dR,eInfo)]
    #find the largest and smallest components along each axis  
    siteMax=[[[ max(c) for c in zip(*aSite)] for aSite in aStrand] for aStrand in siteXYZ]
    siteMin=[[[ min(c) for c in zip(*aSite)] for aSite in aStrand] for aStrand in siteXYZ]
    #find the distance between the outermost components along each axis
    abc=[[[ cMax-cMin for (cMin,cMax) in zip(aMin,aMax)] for (aMin,aMax) in zip(minStrand,maxStrand)] for (minStrand,maxStrand) in zip(siteMin,siteMax)]
    
    #group output data and compute relative ellipsoid lengths from principle 
    #moments of inertia scaled to the distance of the outermost components
    ellipsoids=[[[computeEllipsoids(c, b, e[0], e[1])] for (c,b,e) in zip(cBB,bBB,eBB)] for (cBB,bBB,eBB) in zip(COG, abc, eInfo)]
    
    return checkEllipsoids(ellipsoids)
    
def swapListElements(listV,index1,index2):
    temp=listV[index1].copy()
    listV[index1]=listV[index2].copy()
    listV[index2]=temp.copy()
    return listV

def computeEllipsoids(COG,size,eVals,eVecs):
    #figure out which axis is the longest axis. This will be the y or b axis.
    yAxis=size.index(max(size))
    #swap the ordering of the axes
    size=swapListElements(size,yAxis,1)
    eVals=swapListElements(eVals,yAxis,1)
    eVecs=transpose(swapListElements(transpose(eVecs),yAxis,1))
    #Figure out which is the smallest axis - should end up as the c/z axis
    zAxis=size.index(min(size))
    #place the xAxis in at 0 (and, automatically, the z axis in 2)
    size=swapListElements(size,2,zAxis)
    eVecs=transpose(swapListElements(transpose(eVecs),2,zAxis))
    eVals=swapListElements(eVals,2,zAxis)
    #Now compute the lengths of the ellipsoid required from the inertial 
    #tensor evalues scaled according to the longest length.
    lengths=lengthsFromEvals(eVals,size,1)
    
    #make sure that the rot matrices are det +1
    eVecs=checkSign(eVecs)
    
    
    return [COG, lengths, eVecs]

#This function does only one thing. It flips the signs of
#the vectors in the rotation matrix to make
#sure they align with the corresponding vectors in
#all groups of sites that must be summed.
def checkEllipsoids(ellipsoids):
    
    numBBs=len(ellipsoids)
    numSites=len(ellipsoids[0])
    
    #loop through each site and compare rmats from successive BBS
    for currSite in range(numSites):
        for curBB in range(numBBs-1):
            #pull out the rmat for the current BB
            rmatOld=ellipsoids[curBB][currSite][0][2]
            #pull out the rmat for the next building block
            curRmat=ellipsoids[curBB+1][currSite][0][2]
            
            #check that projection of the next rmat x, y and z 
            #on the current BB is +ve. if it isn't then flip the sign 
            #of the next BB's rmat.
            if dot(array(transpose(curRmat))[0],array(transpose(rmatOld))[0])<0:
                curRmat[0,0]*=-1.0
                curRmat[1,0]*=-1.0
                curRmat[2,0]*=-1.0

            if dot(array(transpose(curRmat))[1],array(transpose(rmatOld))[1])<0:
                curRmat[0,1]=curRmat[0,1]*-1.0
                curRmat[1,1]*=-1.0
                curRmat[2,1]*=-1.0

            if dot(array(transpose(curRmat))[2],array(transpose(rmatOld))[2])<0:
                curRmat[0,2]*=-1.0
                curRmat[1,2]*=-1.0
                curRmat[2,2]*=-1.0
    
    return ellipsoids


def checkSign(e):
    #print "visitE"
    if abs(linalg.det(e)+1)<1e-9:
        #print "swapped sign"
#        print 'e:',e
#        print 'det(e):',linalg.det(e)
        x=array(transpose(e)[0])[0]
        y=array(transpose(e)[1])[0]
        z=cross(x,y)
        outE=transpose(matrix([x,y,z]))
#        print 'det(outE):',linalg.det(outE)
#        print 'outE:',outE
    else:
        outE=e
    return outE

#data supplied to the function as the result of linalg.eig() operating on a matrix.
#i.e.data=(array([e1,e2,e3]), matrix[[u1,v1,w1],[u2,v2,w2],[u3,v3,w3]]) 
#The eigenvectors u=(u1,u2,u3), v=(v1,v2,v3) and w=(w1,w2,w3) listed in column format.
#the function figures out the order the eigen values need to be in 
#to rank them in the order middle, smallest, largest
#and then sorts the eigenvalues returning them as an array,
#and then applys the same sort to the eigenvectors returning them as a matrix 
#with the new ordering of the eigenvectors in column format.
#Thus the data is returned in the format:
#i.e.data=(array([e1',e2',e3']), matrix[[u1',v1',w1'],[u2',v2',w2'],[u3',v3',w3']]) 
#The eigenvectors u'=(u1',u2',u3'), v'=(v1',v2',v3') and w'=(w1',w2',w3') 
#listed in column format.
def rank(data):
    #extract evals from data
    evals=data[0].tolist()
    #find min and max evals - note that the min eval corresponds to maxIndex
    maxIndex=evals.index(min(evals))
    minIndex=evals.index(max(evals))
    #construct two sets and subtract them to figure out which one is the third index
    maxMinSet=set([minIndex,maxIndex])
    allSet=set([0,1,2])
    medIndex=list(allSet-maxMinSet)
    #construct a list of indices in the order than we want the evals and vectors.
    indices=[medIndex[0], maxIndex, minIndex]   
    #sort evalues and return as an array
    rankedEvals=array([data[0][i] for i in indices])
    #sort evectors in same manner and return as an array
    rankedEvecs=transpose(matrix([[val for sublist in data[1][:,i].tolist() for val in sublist] for i in indices]))
    #construct return vector as a tuple.
    retVal=(rankedEvals,rankedEvecs)
    return retVal


def lengthsFromEvals(E,size,lAxis):
    #we assume a uniform density of atoms in a beta strand.
    #Thus the overall moment of inertia can be used to estimate the geometric
    #size of the beta strand, if we make some assumptions about the shape of the beta strand.
    #we assume it is more like an elliptical cylinder than an ellipse, 
    #so can reverse equations for geometric moment of inertia 
    #for an elliptical cylinder to estimate relative dimensions of beta strand
    #The smallest moment of inertia corresponds to rotations about the longest axis of the ellipsoid.
    #Could also assume an ellipsoidal shape but the dimensions of the ellipsoid 
    #that yields the same moments of inertia as the atomic distribution 
    #results in a much larger ellipsoid than is physically realistic.
    lengths=[0,0,0,0,0,0]
    #E[1] is always the smallest eigenvalue so estimate size a,b,c

    #compute the relative sizes of the repulsive ellipsoid
    if lAxis==0:
        lengths[0]= 0.5*sqrt(-6*E[0]+6*E[1]+6*E[2])
        lengths[1]= sqrt(2*E[0]-2*E[1]+2*E[2])
        lengths[2]= sqrt(2*E[0]+2*E[1]-2*E[2])

    if lAxis==1:
        lengths[0]= sqrt(-2*E[0]+2*E[1]+2*E[2])
        lengths[1]= 0.5*sqrt(6*E[0]-6*E[1]+6*E[2])
        lengths[2]= sqrt(2*E[0]+2*E[1]-2*E[2])

    if lAxis==2:
        lengths[0]= sqrt(-2*E[0]+2*E[1]+2*E[2])
        lengths[1]= sqrt(2*E[0]-2*E[1]+2*E[2])
        lengths[2]= 0.5*sqrt(6*E[0]+6*E[1]-6*E[2])

    #make the attractive size the same.
    lengths[3]=lengths[0]
    lengths[4]=lengths[1]
    lengths[5]=lengths[2]
    #Having established the relative ratios of the axes of the ellipse which gives a reasonable
    #approximation to the distribution of the atoms in the beta strands, it's a question
    #of setting the scale.
    #Normalise the largest length to one and multiply by length of longest axis from inertial analysis
    maxL=max(lengths)
    lengths=0.9*size[lAxis]*array(lengths)/(maxL)
    #lengths returned are 2x the semi-axis lengths of the ellipsoid.
    return lengths

#wrapper for conversion function to correctly manage lists of sites.
def convertSitesXYZToRMat(sitesXYZ):
    return [[ siteXYZToRMat(site) for site in strand] for strand in sitesXYZ]
def siteXYZToRMat(siteXYZ):
    #Convert a site from an xyz representation (7 spheres) back to a single ellipsoid
    aa=siteXYZ[0][1][3]
    ba=siteXYZ[0][1][4]
    ca=siteXYZ[0][1][5]
    #extract COG
    COG=siteXYZ[0][0]
    #compute lengths of ellipsoid from positions vectors of end points.
    a=sqrt(vdot(siteXYZ[1][0]-COG,siteXYZ[1][0]-COG))
    b=sqrt(vdot(siteXYZ[3][0]-COG,siteXYZ[3][0]-COG))
    c=sqrt(vdot(siteXYZ[5][0]-COG,siteXYZ[5][0]-COG))
    #reconstruct rotation matrix from principle vectors (use +ve x,y and z sites)
    rMat1=(siteXYZ[2][0]-COG)/a
    rMat2=(siteXYZ[4][0]-COG)/b
    rMat3=(siteXYZ[6][0]-COG)/c
    #format rotation matric correctly
    RMat=matrix([[ rMat1[0], rMat2[0], rMat3[0]],[rMat1[1], rMat2[1], rMat3[1]],[rMat1[2], rMat2[2], rMat3[2]]])
    #construct correct format output array
    siteRMat=[[COG,array([2.0*a,2.0*b,2.0*c,aa,ba,ca]),RMat]]
    return siteRMat

#wrapper to correctly manage lists of sites
def convertSitesRMatToXYZ(sitesRMat):
    return [[ siteRMatToXYZ(site) for site in strand] for strand in sitesRMat]
def siteRMatToXYZ(siteRMat):
    #convert a site from a single anisotropic ellipsoid to an array of seven spheres to capture the size position and anisotropy of the ellipsoid.
#use seven rather than four to cope with roTATIONS.
    aa=siteRMat[0][1][3]
    ba=siteRMat[0][1][4]
    ca=siteRMat[0][1][5]
    #extract COG of ellipsoid
    COG=siteRMat[0][0]
    #extract rotation matrix
    rmat=siteRMat[0][2]
    #compute principal vectors in the ellipsoid frame, scaled to the size of the ellipsoid.
    u=array([siteRMat[0][1][0]/2.0,0,0])
    v=array([0, siteRMat[0][1][1]/2.0, 0])
    w=matrix([0,0,siteRMat[0][1][2]/2.0])
    #find principal direction vectors of ellipsoid in lab frame, 
    uNew=array(u*transpose(rmat))
    vNew=array(v*transpose(rmat))
    wNew=array(w*transpose(rmat))
    #export centre of gravity of ellipsoid
    siteXYZ=[[COG,array([1,1,1,aa,ba,ca]),matrix([[1,0,0],[0,1,0],[0,0,1]])]]
    #append to the output array successive structures detailing each principal vectors
    siteXYZ+=[[COG-uNew[0],array([1,1,1,aa,ba,ca]),matrix([[1,0,0],[0,1,0],[0,0,1]])]]
    siteXYZ+=[[COG+uNew[0],array([1,1,1,aa,ba,ca]),matrix([[1,0,0],[0,1,0],[0,0,1]])]]
    siteXYZ+=[[COG-vNew[0],array([1,1,1,aa,ba,ca]),matrix([[1,0,0],[0,1,0],[0,0,1]])]]
    siteXYZ+=[[COG+vNew[0],array([1,1,1,aa,ba,ca]),matrix([[1,0,0],[0,1,0],[0,0,1]])]]
    siteXYZ+=[[COG-wNew[0],array([1,1,1,aa,ba,ca]),matrix([[1,0,0],[0,1,0],[0,0,1]])]]
    siteXYZ+=[[COG+wNew[0],array([1,1,1,aa,ba,ca]),matrix([[1,0,0],[0,1,0],[0,0,1]])]]
    
    #perform a check to ensure sites are always arranged in the correct order for averaging later on.
    #COG is always COG easy.
    #U direction (a direction) - make first one in array the most -ve x.
    #if siteXYZ[1][0][0]>siteXYZ[2][0][0]:
    #    print "swap x"
    #    rmat=transpose(-1.0*(transpose(rmat)[0]))
    #    swapElements(siteXYZ)
        
        
    #    tempXYZ=siteXYZ[1]
    #    siteXYZ[1]=siteXYZ[2]
    #    siteXYZ[2]=tempXYZ
    
    #V direction (b direction) - make first one in array the most -ve y.
    #if siteXYZ[3][0][1]>siteXYZ[4][0][1]:
    #    print "swap y"        
    #    tempXYZ=siteXYZ[3]
    #    siteXYZ[3]=siteXYZ[4]
    #    siteXYZ[4]=tempXYZ
    #W direction (c direction)- make first one in array the most -ve z.
    #if siteXYZ[5][0][2]>siteXYZ[6][0][2]:
    #    print "swap z"
    #    tempXYZ=siteXYZ[5]
    #    siteXYZ[5]=siteXYZ[6]
    #    siteXYZ[6]=tempXYZ
    
    
    return siteXYZ

#takes a single building block and returns all the sites in it as a list of COG-RMAT pairs, dropping the size information
def  convertXYZToCoords(XYZ):
     #flatXYZ=[ BB for sublist in XYZ for BB in sublist]
     flatXYZ=[ sites for sublist in [BB for sublist2 in XYZ for BB in sublist2] for sites in sublist]
     out=[[ellipsoid[0],ellipsoid[2]] for ellipsoid in flatXYZ]
     return out

#takes a set of test coords, pysites and target coords and uses GMIN's
#pysites potential to compute the best alignment between the two sets. 
#Computes the matrix which best aligns the orientation of the start and finish coords
#when their centre of gravities are aligned.
#The output flag controls the return:
# 0: the distance between the optimal alignments is returned
# 1: the distance,the rotation matrix and the aligned coords
# 2: just the aligned coords
# 3: a dump of information for debugging
# 4: a dump of information for debugging called with a different directory structure.
def AlignPYSites(coords,finish,BBRMat,rOn,aOn,output):
    #create a temporary working directory
    try:
        os.chdir('alignPYSites')
    except OSError:
        subprocess.call(['mkdir','alignPYSites'])
        os.chdir('alignPYSites')

    #write a file containing the target positions and orientations of the building blocks
    writeCoords('finish',finish,1)
    #write the current positions and orientation of the building blocks
    writeCoords('coords',coords,1)
    #writePYsites file with the building block information
    writePYSites('pysites.xyz',BBRMat, rOn, aOn)

    numBBs=len(coords)
    f=open('perm.allow','w')
    f.write('0\n')
    f.close()
 
    f=open('data','w')
    f.write('PERMOPT\n')
    f.write('PY 5.0 10.0\n')
    f.write('STEPS 10 1.0\n')
    f.write('DEBUG\n')
    f.write('TEMPERATURE 1.0\n')
    f.close()
    #call and wait for GMIN to finish
    devnull=open('/dev/null','w')
    subprocess.call('GMIN',stderr=devnull,stdout=devnull)
    #find the rotation required to perform alignment and the distance between them
    rMatBest=readRMATBEST('GMIN_out')

    #compute the aligned coords at the centre of gravity of finish
    alignedCoords=performAlignment(coords,finish,BBRMat,rMatBest[1])
 
    if output==0:
       out=rMatBest[0][0]
    if output==1:
       out=[rMatBest,alignedCoords]
    if output==2:
       out=alignedCoords
    if output==3:
      print rMatBest
      startStruc=convolveBBCoords(BBRMat,coords)
      finishStruc=convolveBBCoords(BBRMat,finish)
      alignedStruc=convolveBBCoords(BBRMat,alignedCoords)
      writeXYZ('PYalign.start.xyz',startStruc,rMatBest[0][0])
      writeXYZ('PYalign.target.xyz',finishStruc,rMatBest[0][0])
      writeXYZ('PYalign.aligned.xyz',alignedStruc,rMatBest[0][0])
      subprocess.call(['cp','PYalign.start.xyz','../.'])
      subprocess.call(['cp','PYalign.target.xyz','../.'])
      subprocess.call(['cp','PYalign.aligned.xyz','../.'])
      subprocess.call(['cp','ellipsoid.xyz','../PYalign.ellipsoid.xyz'])
      sys.exit('Deliberate Termination for debugging purposes')
    if output==4:
      print rMatBest
      startStruc=convolveBBCoords(BBRMat,coords)
      finishStruc=convolveBBCoords(BBRMat,finish)
      alignedStruc=convolveBBCoords(BBRMat,alignedCoords)
      writeXYZ('PYalign.start.xyz',startStruc,rMatBest[0][0])
      writeXYZ('PYalign.target.xyz',finishStruc,rMatBest[0][0])
      writeXYZ('PYalign.aligned.xyz',alignedStruc,rMatBest[0][0])
      subprocess.call(['cp','PYalign.start.xyz','../../.'])
      subprocess.call(['cp','PYalign.target.xyz','../../.'])
      subprocess.call(['cp','PYalign.aligned.xyz','../../.'])
      subprocess.call(['cp','ellipsoid.xyz','../../PYalign.ellipsoid.xyz'])
      sys.exit('Deliberate Termination for debugging purposes')
    #clean up
    os.chdir('..')
    shutil.rmtree('alignPYSites')
    return out

#generate a permallow file for the alignment
def genPermAllow(numAtoms,numEPerSite):
    #numatoms per building block
    numAtomsBB=7*numEPerSite
    #number of building blocks
    numBBs=numAtoms/numAtomsBB
    #number of possible swaps
    numSwaps=sum([c for c in range(numBBs)])
    #compute start numbers for swappings
    atomNums=[ [array(range(numAtomsBB))+1+numAtomsBB*bb] for bb in range(numBBs)]

    #print 'numAtoms:',numAtoms
    #print 'numEPerSite:',numEPerSite
    #print 'numAtomsBB:',numAtomsBB
    #print 'numSwaps:',numSwaps
    #print 'atomNums:',atomNums

    #write a perm.allow file; assume we're in the right directory. user responsibility.
    f=open('perm.allow','w')
    f.write(str(numSwaps)+'\n')
    for currBB in range(numBBs-1):
       for i in range(numBBs-currBB-1):
          currBB2=i+currBB+1
          #print currBB,currBB2
          f.write('2 '+str(numAtomsBB-1)+'\n')
          bb1Atoms=atomNums[currBB]
          #print bb1Atoms
          bb2Atoms=atomNums[currBB2]
          #print bb2Atoms
          outstr=''
          for (a,b) in zip(bb1Atoms[0],bb2Atoms[0]):
             outstr+=str(a)+' '+str(b)+' '
          f.write(outstr+'\n')
    f.close()
    return 

def writeDataFile():
    #write a data instruction file
    f=open('data','w')
    f.write('PERMOPT\n')
    f.write('STEPS 10 1.0\n')
    f.write('DEBUG\n')
    f.write('TEMPERATURE 1.0\n')
    f.flush()
    f.close()
    return
 
#takes a list of start and finish coords and a building block in xyz format.
#Convolves them and generates a list of spheres which it then aligns.
#Computes the matrix which best aligns the orientation of the start and finish coords
# when their centre of gravities are aligned.
# The output flag controls the return:
# 0: the distance between the optimal alignments is returned
# 1: the distance,the rotation matrix and the aligned coords
# 2: just the aligned coords
# 3: a dump of information for debugging
# 4: a dump of information for debugging called with a different directory structure.
def AlignXYZ(coords,finish,BBXYZTest,BBXYZTarget,output):

    #create a temporary working directory
    try:
        os.chdir('alignXYZ')
    except OSError:
        subprocess.call(['mkdir','alignXYZ'])
        os.chdir('alignXYZ')
    #convolve the structures to generate an xyz list
    target=convolveBBCoords(BBXYZTarget,finish)
    test=convolveBBCoords(BBXYZTest,coords)
    #Flatten the xyz lists to a coords format - one entry for each sphere in a flat list.
    #and drop the ellipsoid size information
    targetCoords=convertXYZToCoords(target)
    testCoords=convertXYZToCoords(test)

    #write a finish file containing only the positions of the target spheres
    writeCoords('finish',targetCoords,0)
    #write a coords file containing only the positions of the aligning spheres
    writeCoords('coords',testCoords, 0)
    #genPermAllow(len(targetCoords),len(target[0]))
    f=open('perm.allow','w')
    f.write('0\n')
    f.flush()
    f.close
    writeDataFile()

    #call and wait for GMIN to finish
    devnull=open('/dev/null','w')
    subprocess.call('GMIN',stderr=devnull,stdout=devnull)

    #find the rotation required to perform alignment and the distance between them
    rMatBest=readRMATBEST('GMIN_out')

    #compute the aligned coords at the centre of gravity of finish
    alignedCoords=performAlignment(coords,finish,BBXYZTest,rMatBest[1])


    if output==0:
       out=rMatBest[0]
    if (output==1)or(output==5):
       out=[rMatBest,alignedCoords]
    if output==2:
       out=alignedCoords
    if output==3:
      print rMatBest
      BBRMatCoords=convertSitesXYZToRMat(BBXYZCoords)
      BBRMatTarget=convertSitesXYZToRMat(BBXYZTarget)
      startStrucR=convolveBBCoords(BBRMatCoords,coords)
      finishStrucR=convolveBBCoords(BBRMatTarget,finish)
      alignedStrucR=convolveBBCoords(BBRMatCoords,alignedCoords)
      startStruc=convolveBBCoords(BBXYZCoords,coords)
      finishStruc=convolveBBCoords(BBXYZTarget,finish)
      alignedStruc=convolveBBCoords(BBXYZCoords,alignedCoords)
      writeXYZ('XYZalign.start.r.xyz',startStrucR,rMatBest[0][0])
      writeXYZ('XYZalign.target.r.xyz',finishStrucR,rMatBest[0][0])
      writeXYZ('XYZalign.aligned.r.xyz',alignedStrucR,rMatBest[0][0])
      writeXYZ('XYZalign.start.xyz',startStruc,rMatBest[0][0])
      writeXYZ('XYZalign.target.xyz',finishStruc,rMatBest[0][0])
      writeXYZ('XYZalign.aligned.xyz',alignedStruc,rMatBest[0][0])
      subprocess.call(['cp','XYZalign.start.r.xyz','../.'])
      subprocess.call(['cp','XYZalign.target.r.xyz','../.'])
      subprocess.call(['cp','XYZalign.aligned.r.xyz','../.'])
      subprocess.call(['cp','XYZalign.start.xyz','../.'])
      subprocess.call(['cp','XYZalign.target.xyz','../.'])
      subprocess.call(['cp','XYZalign.aligned.xyz','../.'])
      subprocess.call(['lowToXYZ.py', 'lowest'])
      subprocess.call(['cp','lowest.xyz','../XYZalign.lowest.xyz'])
      sys.exit('Deliberate Termination for debugging purposes')
    if output==4:
      print rMatBest
      BBRMatCoords=convertSitesXYZToRMat(BBXYZCoords)
      BBRMatTarget=convertSitesXYZToRMat(BBXYZTarget)
      startStrucR=convolveBBCoords(BBRMatCoords,coords)
      finishStrucR=convolveBBCoords(BBRMatTarget,finish)
      alignedStrucR=convolveBBCoords(BBRMatCoords,alignedCoords)
      startStruc=convolveBBCoords(BBXYZCoords,coords)
      finishStruc=convolveBBCoords(BBXYZTarget,finish)
      alignedStruc=convolveBBCoords(BBXYZCoords,alignedCoords)
      writeXYZ('XYZalign.start.r.xyz',startStrucR,rMatBest[0][0])
      writeXYZ('XYZalign.target.r.xyz',finishStrucR,rMatBest[0][0])
      writeXYZ('XYZalign.aligned.r.xyz',alignedStrucR,rMatBest[0][0])
      writeXYZ('XYZalign.start.xyz',startStruc,rMatBest[0][0])
      writeXYZ('XYZalign.target.xyz',finishStruc,rMatBest[0][0])
      writeXYZ('XYZalign.aligned.xyz',alignedStruc,rMatBest[0][0])
      subprocess.call(['cp','XYZalign.start.r.xyz','../../.'])
      subprocess.call(['cp','XYZalign.target.r.xyz','../../.'])
      subprocess.call(['cp','XYZalign.aligned.r.xyz','../../.'])
      subprocess.call(['cp','XYZalign.start.xyz','../../.'])
      subprocess.call(['cp','XYZalign.target.xyz','../../.'])
      subprocess.call(['cp','XYZalign.aligned.xyz','../../.'])
      subprocess.call(['lowToXYZ.py', 'lowest'])
      subprocess.call(['cp','lowest.xyz','../../XYZalign.lowest.xyz'])
      sys.exit('Deliberate Termination for debugging purposes')
 
    #clean up
    os.chdir('..')
    #shutil.rmtree('alignXYZ')
    return out

#returns a set of aligned coords
def performAlignment(start,finish,BB,rMat):
    #compute the full structure
    startPosns=convolveBBCoords(BB,start)
    finishPosns=convolveBBCoords(BB,finish)
    #compute the full centre of gravity of the structure
    startCOG=computeCOG(startPosns)
    finishCOG=computeCOG(finishPosns)
    #print 'start',start
    #translate the start coords
    startCentered=[[pos[0]-startCOG, pos[1]] for pos in start]
    #print 'startCog',startCOG
    #print 'finishCog',finishCOG
    #print 'startCentered',startCentered   
    #rotate the start coords
    startCenteredRotated=[ globalRotateCoords(pos,rMat) for pos in startCentered]
    #translate to the finish COG and return
    return [[pos[0]+finishCOG, pos[1]] for pos in startCenteredRotated]


#Align two building blocks using xyz method. BBs are input as XYZ lists of length 1.
#Return BBTest Aligned with BBFinal
#Here we assume the BBs are already aligned at their COG at the origin.
def AlignBB(BBTest,BBTarget):
    #create a temporary working directory
    try:
        os.chdir('alignBB')
    except OSError:
        subprocess.call(['mkdir','alignBB'])
        os.chdir('alignBB')
    #Flatten the BBs to a coords format - one entry for each sphere in a flat list.
    #and drop the ellipsoid size information
    targetCoords=convertXYZToCoords(BBTarget)
    testCoords=convertXYZToCoords(BBTest)

    #get the number of ellipsoids per site
    numEPerSite=len(BBTarget[0])
    #compute number of atoms
    numAtoms=len(targetCoords)
    #write a finish file containing only the positions of the target spheres
    writeCoords('finish',targetCoords,0)
    writeCoords('coords',testCoords, 0)
    #write a perm.allow file
    f=open('perm.allow','w')
    f.write('0\n')
    #f.write('1 0\n')
    #b=array(range(numAtoms))+1
    #for atom in b:
    #   f.write(str(atom)+' ')
    #f.write('\n')
    f.close()
    #write a data instruction file
    writeDataFile()

    #call and wait for GMIN to finish
    devnull=open('/dev/null','w')
    subprocess.call('GMIN',stderr=devnull,stdout=devnull)

    #find the rotation required to perform alignment and the distance between them
    rMatBest=readRMATBEST('GMIN_out')

    #compute the aligned coords at the centre of gravity of finish
    BBAlign=[[[[globalRotateSites(E,rMatBest[1]) for E in site] for site in BB] for BB in BBTest],rMatBest]
  
    #clean up
    os.chdir('..')
    shutil.rmtree('alignBB')
    return BBAlign


def computeCOG(siteList):
    #first flatten this list
    flatList=convertXYZToCoords(siteList)
    #compute cog
    return array(sum([pos[0] for pos in flatList],axis=0)/len(flatList))



def readLowest(filename):
    """Parses lowest file """
    #open file
    f = open(filename, 'r')
    #read in data
    inp=f.readlines()
    numAtoms=int(inp[0])
    lenMin=2+numAtoms
    reshapeInp=[[inp[i+j*(lenMin)] for i in range(lenMin)] for j in range(len(inp)/(lenMin))]
    f.close()
    return reshapeInp

#reads lowest style file and splits the entries up into chunks of 7
#These are organised into groups of numEPerSite and then returned in one big list.
def lowestToXYZ(lowest,numEPerSite):
    numEllipsoids=(len(lowest)-2)/7
    data=[[lowest[2+i+j*7].split() for i in range(7)] for j in range(numEllipsoids)]
    ellipsoids=[[[array([float(ell[1]),float(ell[2]),float(ell[3])]),array([1,1,1,1,1,1]),matrix([[1,0,0],[0,1,0],[0,0,1]])] for ell in site] for site in data]
    return [[e1,e2] for (e1,e2) in zip(ellipsoids[0::2],ellipsoids[1::2])]

def readRMATBEST(filename):
    f=open(filename,'r')
    data=f.readlines()
    GMIN_out=[ line.split() for line in data]
    sRMatIndex=[GMIN_out.index(x) for x in GMIN_out if x[0] in ['RMATBEST:']]
    vals=GMIN_out[sRMatIndex[0]+1:sRMatIndex[0]+4]
    rMat =transpose(matrix([ [float(vals[0][0]),float(vals[0][1]),float(vals[0][2])], [float(vals[1][0]),float(vals[1][1]),float(vals[1][2])], [float(vals[2][0]),float(vals[2][1]),float(vals[2][2])]]))
    distance=float(GMIN_out[sRMatIndex[0]+4][3])
    f.close()
    return [distance,rMat]

def averageBB(BBList):
    #compute average building block
    #loop over sites; just using regular for loops because I can't be bothered to work out
    #a pythonic method. There's probably a sexy one liner
    numBBs=len(BBList)
    numSitesPerBB=len(BBList[0])
    numEllPerSite=len(BBList[0][0])
    BB=[[None for _ in range(numEllPerSite)] for _ in range(numSitesPerBB)]
    #testi=0
    #testj=6
    #testk=1
    #print "BB0"
    #print BBList[0][testi][testj][testk]
    #print "BB1"
    #print BBList[1][testi][testj][testk]
    #print "BB2"
    #print BBList[2][testi][testj][testk]
    #print "BB3"
    #print BBList[3][testi][testj][testk]
    #print "BB4"
    #print BBList[4][testi][testj][testk]
    


    #objective is to compute the average position of each sphere in the xyz representation.
    #Then convert this to an rmat ellipsoid.
    #Assumes that each sphere corresponds to the equivalent sphere in the 
    #other buildig blocks.
    for j in range(numSitesPerBB):
         for k in range(numEllPerSite):
             for i in range(numBBs):
                 if i==0:
                     #if this is the first ellipsoid prime the sum
                     BB[j][k]=BBList[i][j][k]
                 else:
                     #add the xy-z ordinates of each corresponding ellipsoid to a running sum
                     # sum runs over the i index which is the building block index
                     BB[j][k][0]+=BBList[i][j][k][0]
                     BB[j][k][1]+=BBList[i][j][k][1]
                     BB[j][k][2]+=BBList[i][j][k][2]
             BB[j][k][0]/=float(numBBs)
             BB[j][k][1]/=float(numBBs)
             BB[j][k][2]/=float(numBBs)
    #convert strand averaged building block to RMat format; add outerlist to make it work.
    #print "BB",BB
    BBRMat=convertSitesXYZToRMat([BB])
    #print "average"
    #print BB[testi][testj][testk]
    
    #find array sizes
    numSitesPerBB=len(BBRMat[0])
    numEllPerSite=len(BBRMat[0][0])
    #average over all the ellipsoids in the building block to make them all the same size.
    for j in range(numSitesPerBB):
        for k in range(numEllPerSite):
            if (k==0)and(j==0):
                 ellSize=BBRMat[0][j][k][1]
            else:
                 ellSize+=BBRMat[0][j][k][1]
    ellSize=ellSize/float(numSitesPerBB*numEllPerSite)
    #set all the ellipse size sub arrays to the same sub array
    BBOut=[[[[curEl[0],ellSize,curEl[2]] for curEl in site] for site in curBB] for curBB in BBRMat]
    #BBOut=[[[[curEl[0],curEl[1],curEl[2]] for curEl in site] for site in curBB] for curBB in BBRMat]
    return BBOut[0]

#defines a basis based on the first ellipsoid and line of centres.
#returns the co-ordinates and orientations of each ellipsoid in the building block in that
#basis.
def zeroBBRotation(BB):
    #Take LOC betwen first pair of ellipsoids as x-axis of basis
    xHat=(BB[1][0][0]-BB[0][0][0])/linalg.norm(BB[1][0][0]-BB[0][0][0])
    #find unit b axis of first ellipsoid
    b1Hat=array(transpose(BB[0][0][2][:,1]))/linalg.norm(BB[0][0][2][:,1])
    b1Hat=b1Hat[0]
    #find z axis of new basis (cross product of xHat and b1)
    zHat=cross(xHat,b1Hat)/linalg.norm(cross(xHat,b1Hat)) 
    #yHat is zHat x xHat
    yHat=cross(zHat,xHat)/linalg.norm(cross(zHat,xHat))
    #Now construct rotation matrix from direction cosines
    RMat=matrix([[xHat[0],yHat[0],zHat[0]],[xHat[1],yHat[1],zHat[1]],[xHat[2],yHat[2],zHat[2]]])
    #now project position vectors and principal vectors of each ellipsoid onto new basis.
    ZeroBB=[[ [array(E[0]*RMat)[0], E[1], transpose(RMat)*E[2]] for E in site] for site in BB]
    return(ZeroBB)


def computeBB(BBList):
    #This function takes a list of BBs. Each BB consists of a list of sites.
    #Each site consists of a list of seven spheres arranged in an ellipsoid.
    #Each BB is colocated and then aligned in turn with the first BB.
    #The average positions and orientation of each site across the aligned superset 
    #is found as well as the average ellipse size.
    #A basis within the prototypical building block is computed and the 
    #and the position ond orientation, with respect to that basis,
    #of each ellipsoid in building block is returned as a single RMat building block.

    #This defines a zero orientation for the building block.

    #First compute the centre of gravity of each building block across the sites
    BBCOG = [sum(SiteCOGList,0)/len(SiteCOGList) for SiteCOGList in [[site[0][0] for site in BBOuter ]for BBOuter in BBList]]
    #translate COG of each building block to the origin.
    ZeroedBBList = [[[ [ellipsoid[0]-COG,ellipsoid[1],ellipsoid[2]]  for ellipsoid in sites] for sites in BB] for (BB,COG) in zip(BBList,BBCOG)]
    #compute the middle most site to compare each building block with
    midSite=int(floor(len(BBList)/2.0))
    #align each building block with the first building block and build a list of rotated building blocks
    AlignedBBList=[ AlignBB([BB],[ZeroedBBList[midSite]])[0][0] for BB in ZeroedBBList]

    #convert aligned BBs to RMat BBs - for output.
    AlignedBBRMat=convertSitesXYZToRMat(AlignedBBList)

    #write aligned data to xyz file for debugging and testing
    writeXYZ('alignedBBs.rmat.xyz',AlignedBBRMat,0.0)
    writeXYZ('alignedBBs.xyz.xyz',AlignedBBList,0.0)

    #Now compute the average BB over the Aligned BBs - use the XYZ representation
    #averageBB returns an RMAT representation.
    BBAverage=averageBB(AlignedBBList)

    # re-orient the building block to it's zero orientation position. This is arbitrary 
    # and amounts to a global orientation of the over all structure.
    ZeroBB=zeroBBRotation(BBAverage)
    
    #returns a single BB in xyz format
    return [ZeroBB] #RMat representation

def computeCoords(refBB,BBList):
    #compute the centre of gravity of all the ellipsoids in each building block.
    BBCOG = [sum(curBB,0)/len(curBB) for curBB in [[site[0][0] for site in BBOuter ]for BBOuter in BBList]]
    Zeroed = [[[ [eSite[0]-COG,eSite[1],eSite[2]]  for eSite in eSites] for eSites in curBB] for (curBB,COG) in zip(BBList,BBCOG)]
    #find the rotation matrix required to align the two blocks
    BBRMatList=[AlignBB([refBB],[curBB])[1][1] for curBB in Zeroed]
    return [[COG,rmat] for (COG, rmat) in zip(BBCOG,BBRMatList)]

def convolveBBCoords(BB,coords):
    #compute COG of each ellipsoid in output
    #printOutput('coords',coords,2)
    #printOutput('BB',BB,2)
    outCOG=[[[ curC[0]+array(ellipsoid[0]*transpose(curC[1]))[0] for ellipsoid in sites] for sites in BB] for curC in coords]
    #copy dimensions of ellipsoid in output
    outSize=[[[ellipsoid[1] for ellipsoid in sites] for sites in BB] for curC in coords]
    #compute orientation of each ellipsoid in output
    outRot=[[[ curC[1]*ellipsoid[2] for ellipsoid in sites] for sites in BB] for curC in coords]
    #concatenate outputs appropriately
    outAll=[[[ [COG,SIZE,ROT] for (COG,SIZE,ROT) in zip(ECOG, ESIZE, EROT)] for (ECOG, ESIZE, EROT) in zip(ECOGS,ESIZES,EROTS)] for (ECOGS,ESIZES,EROTS) in zip(outCOG,outSize,outRot)]
    return outAll

def readPairs(filename):
    """Parses pairs file assuming n m'"""
    f = open(filename, 'r')
    return [array([int(x) for x in line.split()]) for line in f.readlines()]
    
def analyseGeometry(BB,filename):
    
    pairs=readPairs(filename)
    
    #compute line of centres between pairs of ellipsoids as specified in the pairs filename
    modelAngles=[]
    for pair in pairs:
        for (site0,site1) in zip(BB[pair[0]],BB[pair[1]]):
            modelAngles+=[[pair,computeAngles(site0,site1)]]
             
    return modelAngles

def computeAngles(E1,E2):

    #Compute Line of Centres
    L=E2[0]-E1[0]
    #compute norm of Line of centres
    D=linalg.norm(L)
    X2=L/D
    Z2=cross(X2, transpose(E1[2])[1])[0] 
    D2=linalg.norm(Z2)
    Z2=Z2/D2
    Y2=cross(Z2,X2)
    D2=linalg.norm(Y2)
    Y2=Y2/D2
    V1=transpose(E1[2])[0]
    V2=transpose(E1[2])[1]
    V3=transpose(E1[2])[2]
    R1=matrix([[dot(V1,X2)[0,0],dot(V2,X2)[0,0],dot(V3,X2)[0,0]],[dot(V1,Y2)[0,0],dot(V2,Y2)[0,0],dot(V3,Y2)[0,0]],[dot(V1,Z2)[0,0],dot(V2,Z2)[0,0],dot(V3,Z2)[0,0]]])
    V1=transpose(E2[2])[0]
    V2=transpose(E2[2])[1]
    V3=transpose(E2[2])[2]
    R2=matrix([[dot(V1,X2)[0,0],dot(V2,X2)[0,0],dot(V3,X2)[0,0]],[dot(V1,Y2)[0,0],dot(V2,Y2)[0,0],dot(V3,Y2)[0,0]],[dot(V1,Z2)[0,0],dot(V2,Z2)[0,0],dot(V3,Z2)[0,0]]])
    #convert orientation matrices into euler angles in basis frame of give pair
    e1=RMatToEuler(R1)
    e2=RMatToEuler(R2)
    #return in format [[[d, alpha2, beta1, beta2, gamma1, gamma2]]]
    return [D,e2[0], e1[1], e2[1], e1[2],e2[2]]

def writeModel(filename,modelType,BB,angles,rOn,aOn,N):
    #only output angles for first pair of ellipsoids in the first pair of sites
    #NOTE that the angle model in gencoords uses b and db, and g and dg rather than b1,b2
    #and g1 and g2, so convert using formula:  b2=b1+db, g2=g1+dg
    f=open(filename,'w')
    outstr=modelType+' 13 '+str(N)+' '+str(len(BB))+' '+str(rOn)+' '+str(aOn)+'\n'
    f.write(outstr)
    ar=str(BB[0][0][1][0])+'\n'
    br=str(BB[0][0][1][1])+'\n'
    cr=str(BB[0][0][1][2])+'\n'
    aa=str(BB[0][0][1][3])+'\n'
    ba=str(BB[0][0][1][4])+'\n'
    ca=str(BB[0][0][1][5])+'\n'
    D=str(angles[0])+'\n'
    alpha=str(angles[1])+'\n'
    beta1=str(angles[2])+'\n'
    deltabeta=str(angles[3]-angles[2])+'\n'
    gamma1=str(angles[4])+'\n'
    deltagamma=str(angles[5]-angles[4])+'\n'
    f.write(ar)
    f.write(br)
    f.write(cr)
    f.write(aa)
    f.write(ba)
    f.write(ca)
    f.write(D)
    f.write(alpha)
    f.write(beta1)
    f.write(deltabeta)
    f.write(gamma1)
    f.write(deltagamma)
    f.write('1.2')
    return

def generateStartCoords(N,vSep):
    return [[array([0.0,0.0,-i*vSep]),array(aaxis_to_mat(array([0.0,0.0,2*pi])))] for i in range(N)]

def optimiseParams(BBRMatTarget,targetCoords,rOn,aOn,output):
    """Find potential values that minimize RMSD"""

    try:
        os.chdir('dump')
        os.chdir('..')
    except OSError:
        subprocess.call(['mkdir','dump'])
         
    #create a temporary working directory
    try:
        os.chdir('optimiseBB')
    except OSError:
        subprocess.call(['mkdir','optimiseBB'])
        os.chdir('optimiseBB')
 
    #write a data file - use same data file for every parameter trial
    f=open('data','w')
    f.write('PYOVERLAPTHRESH 1.0 0.1\n')
    f.write('SLOPPYCONV 1.0D-5\n')
    f.write('UPDATES 1000\n')
    f.write('MAXERISE 1.0D-4\n')
    f.write('PY 0.19D0 1.0D0\n')
    f.write('TIGHTCONV 1.0D-8\n')
    f.write('EDIFF 1.0D-2\n') 
    f.write('MAXIT 5000 5000\n')
    f.write('STEPS 0 1.0\n')
    f.write('STEP 0.1 0.0 0.1 0\n')
    f.write('MAXBFGS 0.1\n')
    f.write('DEBUG\n')
    f.write('RADIUS 100.0\n')
    f.close()


    #We need to convert the scale to reduced units in which the repulsive length of the first ellipsoid is always 1.
    #To do this we need to scale the target and building block coords as well. The orientations do not need to be touched.
    
    #set the scale factor
    scaleF=1.0/BBRMatTarget[0][0][1][1]
    
    #Scale the building block and target Coords to the reduced units.
    scaledInfo=scale(BBRMatTarget, targetCoords, scaleF)
    BBRMatTargetScaled=scaledInfo[0]
    targetCoordsScaled=scaledInfo[1]
    scaledTargetXYZ=convolveBBCoords(BBRMatTargetScaled,targetCoordsScaled)
    writeXYZ('scaledTarget.xyz',scaledTargetXYZ,100.0)
    
    
    print 'initial:'
    params=BBRMatTargetScaled[0][0][1]
    print params
    
    #abeta optimisation data
    #listOfBounds=array([(0.1, 1.0), (0.1, 1.0), (0.02, 1.0), (0.02,1.0), (0.02, 1.0)])
    #paramsToFit=array([0.37564492093, 0.165604710438, 0.0996869501283, 0.130403484626, 0.194509008735])
 
    #hets - 8 ellipsoids 
    listOfBounds=array([(0.5, 0.7), (0.2,0.3) , (0.35,0.45), (0.26, 0.35), (0.2, 0.3)])
    paramsToFit=array([0.60747578035,  0.277400812534, 0.384923146859, 0.310554578617, 0.263646653866])
    print listOfBounds
    print paramsToFit    
    half=paramsToFit/2.0
    print '%.20f' % half[0], '%.20f' % half[1], '%.20f' % half[2], '%.20f' % half[3], '%.20f' % half[4]
    paramOrder=[1,0,3,4,2]
 



    #loop a maximum of maxLoops times
    #on each loop take each parameter in turn and find the value 
    #of the parameter (within the bound) which minimises 
    #the distance from the target structure
    #keep looping until self-co nsistency is attained.
    maxLoops=2 # 500
    numLoops=0
    epsilon=1e-5
    carryOn=1
    #random.shuffle(paramOrder)
    print 'Start Param Order: ',paramOrder

    while carryOn==1 and numLoops<maxLoops:
        #heart beat so user knows whats going on.
        outstr= 'Loop '+str(numLoops)+' of '+str(maxLoops)
        print outstr

        #preserve the last set of parameters to monitor progress
        paramsOld=copy.deepcopy(paramsToFit)

        #minimise each parameter.
        pNum=0
        for param in paramsToFit:
            #get the appropriate limits for the current parameter
            bound=listOfBounds[paramOrder[pNum]]
            #output sanity check

            #create variable storing pointer to function rmsdPot
            fToMinimise = (lambda x: rmsdPot(x, paramOrder[pNum], paramsToFit, BBRMatTargetScaled, targetCoordsScaled,rOn,aOn, output))

            #submit point to f to the bfgs minimiser in scipy which 
            #tries value of x until the
            #response from the function is minimised.
            result_p = scipy.optimize.fmin_l_bfgs_b(fToMinimise, [param], bounds=[bound], factr=1e3, approx_grad=True, epsilon=1e-5)

            #If the minimise returns a cold fusion state as the final answer
            if result_p[1]==1000000.0:
                print 'parameter: '+str(pNum)+' value: '+str(paramsToFit[pNum])
                sys.exit('Cold Fusion Detected')
            else:
                #record the result
                paramsToFit[paramOrder[pNum]]=result_p[0][0]
                #next param
                pNum=pNum+1

                params=array([paramsToFit[0], 1.0, paramsToFit[1], paramsToFit[2], paramsToFit[3], paramsToFit[4]])
                print 'err: ',result_p[1],', params: ',params
            
                #Take the latest parameters and construct the building block in scaled down coords
                BBInterimRMat=[[[E[0],params,E[2]] for E in eSite] for eSite in BBRMatTargetScaled]
                #compute the aligned coords into which the structure assembles
                interimData=rmsdPot(paramsToFit[0],0,paramsToFit,BBRMatTargetScaled,targetCoordsScaled,rOn,aOn,5)
            
                #Scale the structure back up to sensible size
                scaledInfo=scale(BBInterimRMat,interimData[1],1/scaleF)
                BBInterimRMatUnscaled=scaledInfo[0]
                BBInterimCoordsUnscaled=scaledInfo[1]
                #compute XYZ version of RMAT
                BBInterimXYZUnscaled=convertSitesRMatToXYZ([BBInterimRMatUnscaled])[0]
            
                #dump the rescaled current structures to file
                minRMat=convolveBBCoords(BBInterimRMatUnscaled,BBInterimCoordsUnscaled)
                #tarRMat=convolveBBCoords(BBRMatTarget,targetCoords)
                minXYZ=convolveBBCoords(BBInterimXYZUnscaled,BBInterimCoordsUnscaled)
                #tarXYZ=convolveBBCoords(BBXYZTarget,targetCoords)

                filename='../dump/'+str(result_p[1])+'_'+str(1.0/scaleF)+'_'+str(params[0])+'_'+str(params[1])+'_'+str(params[2])+'_'+str(params[3])+'_'+str(params[4])+'_'+str(params[5])
                writeXYZ(filename+'min.xyz.xyz',minXYZ,result_p[1])
                #writeXYZ(filename+'tar.xyz.xyz',tarXYZ,err)
                writeXYZ(filename+'min.rmat.xyz',minRMat,result_p[1])
                #writeXYZ(filename+'tar.rmat.xyz',tarRMat,err)
 
        #When all the parameters are done compute the difference 
        #with the old ones.
        deltaParams=paramsToFit-paramsOld
        print ['paramsToFit',paramsToFit]
        print ['paramsOld',paramsOld]
        print ['delta params: ',deltaParams]
        #assume we aren't carrying on and the results are good.
        carryOn=0
        #if one of the parameters is more then epislon from last params.
        #carry on.
        for a in deltaParams:
            if (carryOn==0)and(abs(a)>epsilon):
                #random.shuffle(paramOrder)
                print 'new order for next loop: ',paramOrder
                carryOn=1
                numLoops=numLoops+1       

    print 'final params reduced: ',params
    print 'final params expanded: ',params/scaleF

    #update BBRMat with these params
    BBRMatFinal=[[[E[0],params/scaleF,E[2]] for E in eSite] for eSite in BBRMatTarget] 

    #generate the final data
    finalDataScaled=rmsdPot(paramsToFit[0],0,paramsToFit,BBRMatTargetScaled,targetCoordsScaled,rOn,aOn,5)

    #unscale the final data
    finalData=[ [fD[0],fD[1]/scaleF] for fD in finalDataScaled] 

    #clean up
    os.chdir('..')
    #shutil.rmtree('optimiseBB')
    return [BBRMatFinal,finalData]

def scale(BBRMat,Coords, scaleF):
    scaleBBRMat=[[ [E[0]*scaleF, E[1]*scaleF, E[2]] for E in Site] for Site in BBRMat]
    scaleCoords=[[C[0]*scaleF,C[1]] for C in Coords]
    return [scaleBBRMat,scaleCoords]

def rmsdPot(x, pNum, params, BBRMatTarget, targetCoords, rOn, aOn, output):
    #set new trial param
    newParams=params
    newParams[pNum]=x
    newNewParams=array([newParams[0],1.0,newParams[1],newParams[2],newParams[3],newParams[4]])
    
    #Create a new building block based on merging new params and old building block
    BBRMatTest=[[[E[0],newNewParams,E[2]] for E in eSite] for eSite in BBRMatTarget]
    
    #writeXYZ('rmattest.rmat.xyz',[BBRMatTest],345.0)
    #xyz=convertSitesRMatToXYZ([BBRMatTest])
    #writeXYZ('rmattest.xyz.xyz',xyz,345.0)
    
    writePYSites('pysites.xyz',BBRMatTest,rOn,aOn)
    #generateStartCoords do this based on the new size of the ellipsoids
    startCoords=targetCoords
    startCoords=generateStartCoords(len(targetCoords),2.4*newNewParams[2])
    
    convolvedCs=convolveBBCoords(BBRMatTest,startCoords)
    writeXYZ('startScaled.xyz',convolvedCs,347.0)
    
    
    #generate the coords file
    writeCoords('coords',startCoords,1)
    #call GMIN process to minimise start coords
    devnull=open('/dev/null','w')
    #call GMIN to minimise structure
    subprocess.call('GMIN',stderr=devnull,stdout=devnull)
    #read in the new building block coords
    minimisedCoords=readCoords('coords.1')
    Energy=checkEnergy()
    if Energy>99999:
        retVal=Energy    
    else:
        #convert building block from rmat to xyz
        BBXYZTest=convertSitesRMatToXYZ([BBRMatTest])[0]
        BBXYZTarget=convertSitesRMatToXYZ([BBRMatTarget])[0]
        #align the minimised structures and target structures 
        #and return the results based on the request in the output flag
        retVal=AlignXYZ(minimisedCoords,targetCoords,BBXYZTest,BBXYZTarget,output)
    
    if output==0:
        err=retVal
        #print 'err: ',err,', params: ',newParams
    if output==1:
        err=retVal[0][0]
        alignedCoords=retVal[1]
        #dump the current structures to file
        minRMat=convolveBBCoords(BBRMatTest,alignedCoords)
        tarRMat=convolveBBCoords(BBRMatTarget,targetCoords)
        minXYZ=convolveBBCoords(BBXYZTest,alignedCoords)
        tarXYZ=convolveBBCoords(BBXYZTarget,targetCoords)

        filename='../dump/'+str(err)+'_'+str(newNewParams[0])+'_'+str(newNewParams[1])+'_'+str(newNewParams[2])+'_'+str(newNewParams[3])+'_'+str(newNewParams[4])+'_'+str(newNewParams[5])
        writeXYZ(filename+'min.xyz.xyz',minXYZ,err)
        #writeXYZ(filename+'tar.xyz.xyz',tarXYZ,err)
        writeXYZ(filename+'min.rmat.xyz',minRMat,err)
        #writeXYZ(filename+'tar.rmat.xyz',tarRMat,err)
        
        print 'err: ',err,', params: ',newNewParams
    if output==5:
        #final call from optimiseParams just return all details
        err=retVal
    
    #print err,newParams

    return err


def checkEnergy():
    coldFusion=0
    f=open('GMIN_out','r')
    data=f.readlines()
    GMIN_out=[ line.split() for line in data]
    FQIndex=[GMIN_out.index(x) for x in GMIN_out if x[0] in ['Final']]
    Energy=float(GMIN_out[max(FQIndex)][4])
    f.close()
    return Energy
    
#utility funciton to printout details of a variable without having to retype the same
#code over, and over, and over, and over again...
def printOutput(name,data,allData):
    print name
    print len(data)
    print len(data[0])
    print len(data[0][0])
    if allData==1:
      print data[0][0]
    if allData==2:
      print data
    return

#construct a rotation matrix by defining one of the principle new axes, and a second vector
#to complete the space using gram-Schmidt variation.
def constructRmat(u1,u2):
    v1Hat=u1/linalg.norm(u1)
    v2=u2-dot(u2,v1Hat)*v1Hat
    v2Hat=v2/linalg.norm(v2)
    v3=cross(v1Hat,v2Hat)
    v3Hat=v3/linalg.norm(v3)    
    return  array([[v1Hat[0],v2Hat[0],v3Hat[0]],[v1Hat[1],v2Hat[1],v3Hat[1]],[v1Hat[2],v2Hat[2],v3Hat[2]]])


#Make a rotation matrix from euler angles using Zxy convention.
#rotation of beta about Z axis, then alpha about line of nodes (body x-axis), and then
#rotation of gamma about body y-axis.
def EulerToRMat(aDeg,bDeg,gDeg):
    #convert from degrees to radians
    a=pi*aDeg/180.0
    b=pi*bDeg/180.0
    g=pi*gDeg/180.0
    #beta rotation about z axis
    #rmat1=matrix([[cos(b),-sin(b),0],[sin(b),cos(b),0 ],[0,0,1]])
    #rotation of alpha about  x body axis
    #rmat2=matrix([[1,0,0],[0,cos(a),-sin(a)],[0, sin(a),cos(a)]])
    #rotation of gamma about y body axis
    #rmat3=matrix([[cos(g),0,-sin(g)],[0,1,0],[sin(g),0,cos(g)]])
    #multiplied in correct order and converted to array
    #rmat=rmat1*(rmat2*rmat3)
    #multiplied out by hand for testing. also copied from genCoords and verfied correct.
    rmat=matrix([[cos(b)*cos(g)+sin(a)*sin(b)*sin(g),  -cos(a)*sin(b), cos(g)*sin(a)*sin(b)-cos(b)*sin(g)],[cos(g)*sin(b)-cos(b)*sin(a)*sin(g)  ,cos(a)*cos(b), -cos(b)*cos(g)*sin(a)-sin(b)*sin(g) ], [ cos(a)*sin(g)   ,sin(a)   ,cos(a)*cos(g) ]])
    return rmat


#computes two sets of euler angles which satisfy the rotation matrix generation requirements.
#This returns possible sets of alphas, betas and gammas to describe the orientation of a given
#ellipsoid or building block.
def RMatToEuler(RMat):
    #check to see if sin(alpha) is +/-1; bz is sin(a).
    if (abs(abs(RMat[2,1])-1.0))>1e-9:
        #if sin(a) is not +/-1 then go ahead; get alpha
        alpha1=arcsin(RMat[2,1])
        alpha2=pi-alpha1

        #compute gamma 
        gamma1=arctan2(RMat[2,0]/cos(alpha1),RMat[2,2]/cos(alpha1))
        gamma2=arctan2(RMat[2,0]/cos(alpha2),RMat[2,2]/cos(alpha2))

        #compute 
        beta1=arctan2(-RMat[0,1]/cos(alpha1),RMat[1,1]/cos(alpha1))
        beta2=arctan2(-RMat[0,1]/cos(alpha2),RMat[1,1]/cos(alpha2))
    else:
        #resolve gimbal lock cases set gamma=0
        if abs(RMat[2,1]-1)<1e-9:
            alpha1=pi/2
            alpha2=pi/2
            gamma1=0.0
            gamma2=-arctan2(RMat[1,0],RMat[0,0])
            beta1=arctan2(RMat[1,0],RMat[0,0])
            beta2=0.0
        else:
            alpha1=-pi/2
            alpha2=-pi/2
            gamma1=0.0
            gamma2=arctan2(-RMat[1,0],-RMat[0,0])
            beta1=arctan2(-RMat[1,0],-RMat[0,0])
            beta2=0.0

    #in general would wrap the angles so they are all in the range -pi to pi.
    #However because of symmetry of ellipsoid, we can map this to -pi/2 to pi/2 wlog.
    #in most cases these makes both solutions identical, but not in all.
    #however, for the most part using small angles on either side of zero sorts us out.
    alpha1=wrap(alpha1,-pi/2.0,pi/2.0)
    alpha2=wrap(alpha2,-pi/2.0,pi/2.0)
    beta1=wrap(beta1,-pi/2.0,pi/2.0)
    beta2=wrap(beta2,-pi/2.0,pi/2.0)
    gamma1=wrap(gamma1,-pi/2.0,pi/2.0)
    gamma2=wrap(gamma2,-pi/2.0,pi/2.0)

    return array([alpha1,beta1,gamma1, alpha2, beta2, gamma2])*180/pi

#wraps a value into the interval
def wrap(angle,lowerLimit,upperLimit):
    interval=upperLimit-lowerLimit
    while (angle<lowerLimit)or(angle>upperLimit):
       if angle<lowerLimit: 
            angle+=interval
       if angle>upperLimit:
            angle-=interval
    return angle

def testWrap():
    angle=array(range(1000))-500
    lowerLimit=-80
    upperLimit=20
    wrappedAngle=[wrap(a,lowerLimit,upperLimit) for a in angle]
    print angle
    print wrappedAngle
    sys.exit()
    return

def testRmatAAxis():
    rmat=EulerToRMat(0,10,34)
    print 'original'
    print rmat
    p=mat_to_aaxis(rmat)
    print 'first p convert'
    print p
    print linalg.norm(p)*180/pi
    rmatConv=aaxis_to_mat(p)
    print 'reconvert back to matrix'
    print rmatConv
    pconv=mat_to_aaxis(rmatConv)
    print 'second p convert'
    print pconv
    print linalg.norm(pconv)*180/pi
    rmatConv2=aaxis_to_mat(pconv)
    print 'second rmat convert'
    print rmatConv2
    sys.exit('end testRMatAAxis')
    return

def testAngle(alpha,beta,gamma,alphaPass,betaPass,gammaPass,f):
    #generate a rotation matrix from some euler angles
    rMat=EulerToRMat(alpha,beta,gamma)
    f.write('rMat:\n')
    f.write(str(rMat))
    f.write('\n')
    #compute the angles which would generate that matrix from the matrix
    angles=RMatToEuler(rMat)
    #use the recomputed angles to generate a matrix
    rMatReconv1=EulerToRMat(angles[0],angles[1],angles[2])
    rMatReconv2=EulerToRMat(angles[3],angles[4],angles[5])
    f.write('rMatConv1:\n')
    f.write(str(rMatReconv1))
    f.write('\n')
    f.write('rMatConv2:\n')
    f.write(str(rMatReconv2))
    f.write('\n')
    #compare the regenerated matrix with theoriginal matrix. Should be the same.
    #if they are the euler angles are good.
    angleOK=0
    if sum(abs(rMatReconv1-rMat)<1e-8):
        angleOK=1
    if sum(abs(rMatReconv2-rMat)<1e-8):
        angleOK=1
    #dump the test details for inspection
    f.write('test: '+str(alpha)+' '+str(beta)+' '+str(gamma)+'\n')
    f.write('pass: '+str(alphaPass)+' '+str(betaPass)+' '+str(gammaPass)+'\n')
    f.write('calc1: '+str(angles[0])+' '+str(angles[1])+' '+str(angles[2])+'\n')
    f.write('calc2: '+str(angles[3])+' '+str(angles[4])+' '+str(angles[5])+'\n')
    f.write('\n')
    return angleOK

def testAngles():
    f=open('test.angles','w')
    angleVals=[180,150,120,90,60,30,0,-30,-60,-90,-120,-150,-180]
    anglePass=[0,-30,-60,90,60,30,0,-30,-60,90,60,30,0]
    for (alpha,alphaPass) in zip(angleVals,anglePass):
        for (beta,betaPass) in zip(angleVals,anglePass):
            for (gamma,gammaPass) in zip(angleVals,anglePass):
                if testAngle(alpha,beta,gamma,alphaPass,betaPass,gammaPass,f)==0:
                    outstr='Angle Test failed  on: '+str(alpha)+' '+str(beta)+' '+str(gamma)
                    sys.exit(outstr)
    f.close()
    sys.exit('Angle Test Passed')
    return

def testPysites():
    alpha1=0 #always leave as zero
    beta1=15
    gamma1=-15
    alpha2=10
    beta2=-20
    gamma2=-15
    gRotA=20
    gRotB=32
    gRotG=10
    gRot=EulerToRMat(gRotA,gRotB,gRotG)
    d=2
    rmat1=EulerToRMat(alpha1,beta1,gamma1)
    p=mat_to_aaxis(rmat1)
    print 'building block ellipsoid 1 test rmat'
    print rmat1
    print p
    rmat2=EulerToRMat(alpha2,beta2,gamma2)
    p=mat_to_aaxis(rmat2)
    print 'building block ellipsoid 2 test rmat'
    print rmat2
    print p 
    #create building block
    BBTest=[[[[array([-d/2,0,0]),array([1,10,3,1,10,3]),rmat1]],[[array([d/2,0,0]),array([1,10,3,1,10,3]),rmat2]]]]
    #output building block as pysites.
    writePYSites('pysites.analysePDB.xyz',BBTest[0],1,1)
    
    BBGeometryPreRot=analyseGeometry(BBTest[0],'2BEG.pair')
    #Rotate the building block by gRot. analyseGeometry should still recover internal angles.
    BBTestGRot=[[ [globalRotateSites(site[0],gRot)] for site in BBTest[0]]]
    
    #analyse building block and recover angles    
    BBGeometryPostRot=analyseGeometry(BBTestGRot[0],'2BEG.pair')
    print 'original'
    print d, alpha2,beta1,beta2,gamma1,gamma2
    print 'calculated pre rot'
    print BBGeometryPreRot
    print 'calculated post rot'
    print BBGeometryPostRot

    #write a model file and call genCoords - generates an output pysites which should be 
    #identical to the one created above
    writeModel('ellipsoid.model','angle',BBTest[0],BBGeometryPreRot[0][1],1,1,1)
    subprocess.call('genCoords')
    sys.exit('end testPYSites and geometry')
    return

def testConvolve():
#function compares the genXYZ.py, convolve and GMIN PY potential for how they convolve 
#building blocks with a coords array and overall rotations.  
#both genXYZ/py and GMIN should work with a pysites file generated either by genCoords 
#or this program.
#Should all give the same results.
#First we generate a single UNrotated building block and output it at the origin in 
#testBB.xyz.
#We then generate a coords arrays and output this as a coords file.
#THen we convolve the internal building block with the coords array and output this as
#convolve - output into convolve.xyz
#Then we write a pysites.xyz file and a data file
#We then call external routines:
#GMIN - with no steps. Out: ellipsoid.xyz, coords.1, GMIN_out
#genXYZ.py - generates coords.xyz file
#Then need to compare testBB.xyz, convolve.xyz, ellipsoid.xyz and coords.xyz
#Should all be the same
    alpha1=0
    beta1=10
    gamma1=-67
    alpha2=45
    beta2=-10
    gamma2=30
    D=2
    BBrotAlpha=15
    BBrotBeta=45
    BBrotGamma=30
    rmat1=EulerToRMat(alpha1,beta1,gamma1)
    p=mat_to_aaxis(rmat1)
    print 'building block ellipsoid 1 rmat'
    print rmat1
    print p
    rmat2=EulerToRMat(alpha2,beta2,gamma2)
    p=mat_to_aaxis(rmat2)
    print 'building block ellipsoid 2 rmat'
    print rmat2
    print p
    rmat3=EulerToRMat(BBrotAlpha,BBrotBeta,BBrotGamma)
    p=mat_to_aaxis(rmat3)
    print 'building block over all rotation rmat'
    print rmat3
    print p
    #create a building block
    BBTest=[[[[array([-D/2,0,0]),array([1,10,3,1,10,3]),rmat1]],[[array([D/2,0,0]),array([1,10,3,1,10,3]),rmat2]]]]
    #output building block
    printOutput('BBTest',BBTest,2)
    #outputbuilding block as if it were an ellipsoid file; doesn't use mat_toaaxis data.
    writeXYZ('testBB.xyz',BBTest,0.0)
    #output building block as pysites - feed into GMIN and genXYZ.
    writePYSites('pysites.xyz',BBTest[0],1,1)
    #create a positions array
    positions=[[array([0,0,0]),rmat3],[array([0,0,2]),rmat3],[array([0,0,4]),rmat3]]
    #generate convolution - uses only rot matrix
    positions2=convolveBBCoords(BBTest[0],positions)
    #write convolution to xyz file directly - avoid using aaxis conversion in plotting.
    writeXYZ('convolve.xyz',positions2,0.0)
    #output the coords file - only outputs angle axis data
    writeCoords('coordsInp',positions,1)
    writeCoords('coords',positions,1)
    #write data file
    f=open('data','w')
    f.write('DEBUG\nPY 5 10 5\nSTEPS 0 1.0\nMAXIT 0.0 0.0\n')
    f.flush() 
    f.close()

    subprocess.call('GMIN')
    subprocess.call(['genXYZ.py','coords','pysites.xyz'])
    subprocess.call(['mv','pysites.xyz','pysites.1.xyz'])
    #test pysites files    
    BBGeometry=analyseGeometry(BBTest[0])
    writeModel('ellipsoid.model','coordsModel',BBTest[0],BBGeometry,1,1,len(positions))
    subprocess.call('genCoords')
    sys.exit('end test convolve')
    return

def globalRotateCoords(C,rMat):
    #Perform a global rotation for an ellipsoid
    return [array(C[0]*transpose(rMat))[0],transpose(transpose(C[1])*transpose(rMat))]
    #return [array(C[0]*transpose(rMat))[0],rMat*C[1M]

def globalRotateSites(E,rMat):
    #sub contract position and orientation rotation to globalRotateCoords
    CRot=globalRotateCoords([E[0],E[2]],rMat)
    #construc output in correct format
    return [CRot[0],E[1],CRot[1]]

def testGlobalRotate():
    alpha1=0
    beta1=0
    gamma1=0
    alpha2=10
    beta2=30
    gamma2=0
    BBrotAlpha=0
    BBrotBeta=0
    BBrotGamma=00
    gRotAlpha=90
    gRotBeta=0
    gRotGamma=0

    rmat1=EulerToRMat(alpha1,beta1,gamma1)
    print 'rmat1'
    print rmat1
    rmat2=EulerToRMat(alpha2,beta2,gamma2)
    print 'rmat2'
    print rmat2
    rmat3=EulerToRMat(BBrotAlpha,BBrotBeta,BBrotGamma)
    print 'rmat3'
    print rmat3
    gRot=EulerToRMat(gRotAlpha,gRotBeta,gRotGamma)
    print 'gRot'
    print gRot
 
    #construct test coords
    testCoords=[[array([0,0,2*i]), rmat3] for i in range(1)]
 
    #construct rotated test coords
    rotatedCoords=[ globalRotateCoords(T,gRot) for T in testCoords]

    #create a building block
    BBRMat=[[[array([-1,0,0]),array([1,10,3,1,10,3]),rmat1]],[[array([1,0,0]),array([1,10,2,1,10,2]),rmat2]]]

    #create a sites list
    testSites=convolveBBCoords(BBRMat,testCoords)
    #create sites list using rotated coords
    convolvedSites=convolveBBCoords(BBRMat,rotatedCoords)
    #rotate the sites list directly
    rotatedSites=[[[globalRotateSites(E,gRot) for E in eSite] for eSite in eSites] for eSites in testSites]
   
    p=mat_to_aaxis(gRot)
    print 'angle_axis:', p
    print linalg.norm(p)*180/pi
    
    writeXYZ('testSites.xyz',testSites,0.0)
    writeXYZ('convolvedSites.xyz',convolvedSites,0.0)
    writeXYZ('rotatedSites.xyz',rotatedSites,0.0)
    sys.exit()

    return


def addNoiseCoords(coords,noise):
    return [ [array([(random.random()-0.5)*noise, (random.random()-0.5)*noise,(random.random()-0.5)*noise])+BB[0],BB[1] ] for BB in coords]

def testPerformAlignment():

    alpha1=0
    beta1=0
    gamma1=0
    alpha2=10
    beta2=30
    gamma2=0
    BBrotAlpha=0
    BBrotBeta=0
    BBrotGamma=45
    gRotAlpha=90
    gRotBeta=0
    gRotGamma=0

    rmat1=EulerToRMat(alpha1,beta1,gamma1)
    print 'rmat1'
    print rmat1
    rmat2=EulerToRMat(alpha2,beta2,gamma2)
    print 'rmat2'
    print rmat2
    rmat3=EulerToRMat(BBrotAlpha,BBrotBeta,BBrotGamma)
    print 'rmat3'
    print rmat3
    gRot=EulerToRMat(gRotAlpha,gRotBeta,gRotGamma)
    print 'gRot'
    print gRot
    rOn=1
    aOn=1
   
    COGOffset1=array([2.4,-1.2,3.3]) 
    COGOffset2=array([0.3, 0.2, -1.2]) 

    #construct target coords
    testCoords=[[array([0,0,2*i]), rmat3] for i in range(5)]
 
    #construct rotated test coords
    targetCoords=[ globalRotateCoords(T,gRot) for T in testCoords]

    #offset target coords
    targetCoordsOffset= [[pos[0]+COGOffset1,pos[1]] for pos in targetCoords]

    #offset testcoords
    testCoordsOffset= [[pos[0]+COGOffset2,pos[1]] for pos in testCoords]

    #create an rmat building block
    BBRMat=[[[array([-1,0,0]),array([1,10,3,1,10,3]),rmat1]],[[array([1,0,0]),array([1,10,2,1,10,2]),rmat2]]]

    #create an xyz building block - dereference at end with [0] so BBXYZ and BBRMat are in sync
    BBXYZ=convertSitesRMatToXYZ([BBRMat])[0]

    #generate aligned coords
    alignedCoordsRMat=performAlignment(testCoordsOffset,targetCoordsOffset,BBRMat,gRot)
    alignedCoordsXYZ=performAlignment(testCoordsOffset,targetCoordsOffset,BBXYZ,gRot)

    #compute aligned coordinates
    targetStrucRMat=convolveBBCoords(BBRMat,targetCoordsOffset)
    testStrucRMat=convolveBBCoords(BBRMat,testCoordsOffset)
    alignedStrucRMat=convolveBBCoords(BBRMat,alignedCoordsRMat)
    targetStrucXYZ=convolveBBCoords(BBXYZ,targetCoordsOffset)
    testStrucXYZ=convolveBBCoords(BBXYZ,testCoordsOffset)
    alignedStrucXYZ=convolveBBCoords(BBXYZ,alignedCoordsRMat)

    writeXYZ('target.rmat.xyz',targetStrucRMat,0.0)
    writeXYZ('test.rmat.xyz',testStrucRMat,0.0)
    writeXYZ('aligned.rmat.xyz',alignedStrucRMat,0.0)
    writeXYZ('target.xyz.xyz',targetStrucXYZ,0.0)
    writeXYZ('test.xyz.xyz',testStrucXYZ,0.0)
    writeXYZ('aligned.xyz.xyz',alignedStrucXYZ,0.0)

    sys.exit('terminate test for checking performing the alignment')
    return

def testAlign():
    alpha1=0
    beta1=0
    gamma1=0
    alpha2=10
    beta2=30
    gamma2=0
    BBrotAlpha=0
    BBrotBeta=0
    BBrotGamma=45
    gRotAlpha=90
    gRotBeta=0
    gRotGamma=0
    noise=.1

    rmat1=EulerToRMat(alpha1,beta1,gamma1)
    print 'rmat1'
    print rmat1
    rmat2=EulerToRMat(alpha2,beta2,gamma2)
    print 'rmat2'
    print rmat2
    rmat3=EulerToRMat(BBrotAlpha,BBrotBeta,BBrotGamma)
    print 'rmat3'
    print rmat3
    gRot=EulerToRMat(gRotAlpha,gRotBeta,gRotGamma)
    print 'gRot'
    print gRot
    rOn=1
    aOn=1
 
    #construct test coords
    testCoords=[[array([0,0,2*i]), rmat3] for i in range(5)]
 
    #construct rotated target coords
    targetCoords=[ globalRotateCoords(T,gRot) for T in testCoords]

    #printOutput('testCoords',testCoords,1)

    #add noise to coords
    testCoords=addNoiseCoords(testCoords,noise)

    #printOutput('testCoordsNoise',testCoords,1)

    #create an rmat building block
    BBRMat=[[[array([-1,0,0]),array([1,10,3,1,10,3]),rmat1]],[[array([1,0,0]),array([1,10,2,1,10,2]),rmat2]]]

    #create an xyz building block - dereference at end with [0] so BBXYZ and BBRMat are in sync
    BBXYZ=convertSitesRMatToXYZ([BBRMat])[0]

    #align the structures using the two different techniques, 
    # output flag 1 returns the distance,the rotation matrix and the aligned coords
    output=1
    alignedXYZRet=AlignXYZ(testCoords,targetCoords,BBXYZ,BBXYZ,output)
    alignedRMatRet=AlignPYSites(testCoords,targetCoords,BBRMat,rOn,aOn,output)


    #convolve the returned coords with the building block used to align them
    alignedXYZ=convolveBBCoords(BBXYZ,alignedXYZRet[1])
    alignedRMat=convolveBBCoords(BBRMat,alignedRMatRet[1])

    #generate xyz data for the target structures
    targetXYZ=convolveBBCoords(BBXYZ,targetCoords)
    targetRMat=convolveBBCoords(BBRMat,targetCoords)

    #generate xyz for the test structures
    testXYZ=convolveBBCoords(BBXYZ,testCoords)
    testRMat=convolveBBCoords(BBRMat,testCoords)

    #ouput structures for perusal
    writeXYZ('aligned.xyz.xyz',alignedXYZ,alignedXYZRet[0][0])
    writeXYZ('target.xyz.xyz',targetXYZ,alignedXYZRet[0][0])
    writeXYZ('test.xyz.xyz',testXYZ,alignedXYZRet[0][0])

    writeXYZ('aligned.rmat.xyz',alignedRMat,alignedRMatRet[0][0])
    writeXYZ('target.rmat.xyz',targetRMat,alignedRMatRet[0][0])
    writeXYZ('test.rmat.xyz',testRMat,alignedRMatRet[0][0])

    rMatR=alignedRMatRet[0][1]
    rMatD=alignedRMatRet[0][0]
    xyzR=alignedXYZRet[0][1]
    xyzD=alignedXYZRet[0][0]

    # in principle should return gRot and distance of zero.
    print 'rMatR:'
    print rMatR
    print 'rMatD:'
    print rMatD
    print 'xyzR:'
    print xyzR
    print 'xyzD:'
    print xyzD
    sys.exit('end testAlign')
    return

def failedAlign():

    filename=getFilename()
    atoms=readPDB(filename+'pdb')
    sitqpe=readSites(filename+'sit')
    sitesRMat=analyseXYZ(site,atoms)
    sitesXYZ=convertSitesRMatToXYZ(sitesRMat)
    writeXYZ(filename+'Analysis.RMat.xyz',sitesRMat,0.0)
    writeXYZ(filename+'Analysis.XYZ.xyz',sitesXYZ,0.0)

    BBRMat=computeBB(sitesXYZ)
    BBXYZ=convertSitesRMatToXYZ(BBRMat)
    writeXYZ(filename+'BB.Rmat.xyz',BBRMat,0.0)
    writeXYZ(filename+'BB.XYZ.xyz',BBXYZ,0.0)
    writePYSites('pysitesDesign.xyz',BBRMat[0],rOn,aOn)

    targetCoords=computeCoords(BBXYZ[0],sitesXYZ)
    writeCoords('finish',targetCoords,1)
    sitesTargetRMat=convolveBBCoords(BBRMat[0],targetCoords)  
    sitesTargetXYZ=convolveBBCoords(BBXYZ[0],targetCoords)  
    writeXYZ(filename+'target.Rmat.xyz',sitesTargetRMat,0.0)
    writeXYZ(filename+'target.XYZ.xyz',sitesTargetXYZ,0.0)

    BBGeometry=analyseGeometry(BBRMat[0])
    writeModel('ellipsoid.model','angle',BBRMat[0],BBGeometry,rOn,aOn,len(targetCoords))
    subprocess.call('genCoords')
 
    #write a data file
    f=open('data','w')
    f.write('CENTRE\n')
    f.write('PYOVERLAPTHRESH 1.0\n')
#    f.write('RANDOMSEED\n')
#    f.write('BHPT 0.1 10.0 0.5\n')
    f.write('SLOPPYCONV 1.0D-5\n')
    f.write('UPDATES 1000\n')
    f.write('MAXERISE 1.0D-4\n')
    f.write('PY 0.5D0 1.0D0 0.5D0\n')
    f.write('TIGHTCONV 1.0D-7\n')
    f.write('PY 5.0D-0 1.0D1 5.0D-0\n')
#    f.write('SAVE 1000\n')
    f.write('EDIFF 1.0D-2\n') 
    f.write('MAXIT 5000 5000\n')
    f.write('STEPS 0 1.0\n')
    f.write('STEP 0.1 0.0 0.1 0\n')
    f.write('MAXBFGS 0.1\n')
    f.write('DEBUG\n')
    f.write('RADIUS 100.0\n')
#    f.write('MPI\n')
    f.close()

    #call GMIN to minimise structure
    subprocess.call('GMIN')
    sys.exit()
    return

def q2aa( qin ):
    """
    quaternion to angle axis
    input Q: quaternion of length 4
    output V: angle axis vector of lenth 3
    """
    q = copy.copy(qin)
    if q[0] < 0.: q = -q
    if q[0] > 1.0: q /= sqrt(dot(q,q))
    theta = 2. * arccos(q[0])
    s = sqrt(1.-q[0]*q[0])
    if s < rot_epsilon:
        p = 2. * q[1:4]
    else:
        p = q[1:4] / s * theta
    return p

def q2mx( qin ):
    """quaternion to rotation matrix"""
    Q = qin / linalg.norm(qin)
    RMX = zeros([3,3], float64)
    Q2Q3 = Q[1]*Q[2];
    Q1Q4 = Q[0]*Q[3];
    Q2Q4 = Q[1]*Q[3];
    Q1Q3 = Q[0]*Q[2];
    Q3Q4 = Q[2]*Q[3];
    Q1Q2 = Q[0]*Q[1];

    RMX[0,0] = 2.*(0.5 - Q[2]*Q[2] - Q[3]*Q[3]);
    RMX[1,1] = 2.*(0.5 - Q[1]*Q[1] - Q[3]*Q[3]);
    RMX[2,2] = 2.*(0.5 - Q[1]*Q[1] - Q[2]*Q[2]);
    RMX[0,1] = 2.*(Q2Q3 - Q1Q4);
    RMX[1,0] = 2.*(Q2Q3 + Q1Q4);
    RMX[0,2] = 2.*(Q2Q4 + Q1Q3);
    RMX[2,0] = 2.*(Q2Q4 - Q1Q3);
    RMX[1,2] = 2.*(Q3Q4 - Q1Q2);
    RMX[2,1] = 2.*(Q3Q4 + Q1Q2);
    return matrix(RMX)

def random_q():
    #uniform random rotation in angle axis formulation
    #input: 3 uniformly distributed random numbers
    #uses the algorithm given in
     #K. Shoemake, Uniform random rotations, Graphics Gems III, pages 124-132. Academic, New York, 1992.
    #This first generates a random rotation in quaternion representation. We should substitute this by
    #a direct angle axis generation, but be careful: the angle of rotation in angle axis representation
    #is NOT uniformly distributed
    u =array([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
    q = zeros(4, float64)
    q[0] = sqrt(1.-u[0]) * sin(2.*pi*u[1])
    q[1] = sqrt(1.-u[0]) * cos(2.*pi*u[1])
    q[2] = sqrt(u[0]) * sin(2.*pi*u[2])
    q[3] = sqrt(u[0]) * cos(2.*pi*u[2])
    return q

def randomRotation():
    q=random_q()
    return q2mx(q)

def testAlignBB():
    alpha1=0
    beta1=10
    gamma1=-67
    alpha2=45
    beta2=-10
    gamma2=30
    alphaG=0
    betaG=20
    gammaG=10
    D=2
    rmat1=EulerToRMat(alpha1,beta1,gamma1)
    rmat2=EulerToRMat(alpha2,beta2,gamma2)
    grot=EulerToRMat(alphaG,betaG,gammaG)

    print 'initial grot:',grot

    #create a building block
    BBTest=[[[[array([-D/2,0,0]),array([1,10,3,1,10,3]),rmat1]],[[array([D/2,0,0]),array([1,10,3,1,10,3]),rmat2]]]]
    grot
 
    #rotate the building block to form a target
    BBTarget=[[[globalRotateSites(E,grot) for E in sites] for sites in BBTest[0]]]
    
    BBTestXYZ=convertSitesRMatToXYZ(BBTest)
    BBTargetXYZ=convertSitesRMatToXYZ(BBTarget)

    #output building block
    writeXYZ('testBB.rmat.xyz',BBTest,0.0)
    writeXYZ('testBB.xyz.xyz',BBTestXYZ,0.0)
    writeXYZ('targetBB.rmat.xyz',BBTarget,0.0)
    writeXYZ('targetBB.xyz.xyz',BBTargetXYZ,0.0)
    BBAligned=AlignBB(BBTestXYZ,BBTargetXYZ)
    writeXYZ('alignedBB.xyz.xyz',BBAligned[0],0.0)
    print 'rmat recovered: ', BBAligned[1]

    sys.exit()
    return
 
def testComputeBB():
    alpha1=0
    beta1=10
    gamma1=-67
    alpha2=45
    beta2=-10
    gamma2=30
    D=2
    rmat1=EulerToRMat(alpha1,beta1,gamma1)
    p=mat_to_aaxis(rmat1)
    print 'building block ellipsoid 1 rmat'
    print rmat1
    print p
    rmat2=EulerToRMat(alpha2,beta2,gamma2)
    p=mat_to_aaxis(rmat2)
    print 'building block ellipsoid 2 rmat'
    print rmat2
    print p
    grot=randomRotation()
    
    #create a known building block
    E1=[array([-D/2,0,0]),array([1,10,3,1,10,3]),randomRotation()]
    E2=[array([D/2,0,0]),array([1,10,3,1,10,3]),randomRotation()]
    Site1=[[E1],[E2]]
    BBTestRot=[Site1]
    BBTest=[zeroBBRotation(BBTestRot[0])]
    #output building block
    printOutput('BBTest',BBTest,2)
    writeXYZ('testBB.xyz',BBTest,0.0)

    #create a positions array
    positions=[[array([0,0,0]),randomRotation()],[array([0,0,4]),randomRotation()],[array([0,0,8]),randomRotation()], [array([0,0,12]),randomRotation()]]
    positionsRot=[ globalRotateCoords(pos,grot) for pos in positions]
    #generate convolution
    positionsRMat=convolveBBCoords(BBTest[0],positionsRot)
    positionsXYZ=convertSitesRMatToXYZ(positionsRMat)
    
    #write to file
    writeXYZ('convolve.RMAT.xyz',positionsRMat,0.0)
    writeXYZ('convolve.xyz',positionsXYZ,0.0)
    #compute the building block by averaging over the stack
    computedBBRMat=computeBB(positionsXYZ)
    alignedData=AlignBB(BBTest,computedBBRMat)
    zeroComputedBBRMat=alignedData[0]
    rMatRecovered=alignedData[1]
    computedBBXYZ=convertSitesRMatToXYZ(zeroComputedBBRMat)
    printOutput('computedBBRMat',zeroComputedBBRMat,2)
    print 'grot: ',grot
    print 'rMatRecovered: ',rMatRecovered
    
    writeXYZ('computedBB.RMat.xyz',zeroComputedBBRMat,0.0)
    writeXYZ('computedBB.xyz',computedBBXYZ,0.0)
    sys.exit()

    return

def testComputeCoords():
    alpha1=0
    beta1=10
    gamma1=-67
    alpha2=45
    beta2=-10
    gamma2=30
    D=2
    rmat1=EulerToRMat(alpha1,beta1,gamma1)
    p=mat_to_aaxis(rmat1)
    print 'building block ellipsoid 1 rmat'
    print rmat1
    print p
    rmat2=EulerToRMat(alpha2,beta2,gamma2)
    p=mat_to_aaxis(rmat2)
    print 'building block ellipsoid 2 rmat'
    print rmat2
    print p
    #create a building block
    BBTest=[[[[array([-D/2,0,0]),array([1,10,3,1,10,3]),rmat1]],[[array([D/2,0,0]),array([1,10,3,1,10,3]),rmat2]]]]
    #output building block
    #create a positions array
    positions=[[array([0,0,0]),randomRotation()],[array([0,0,4]),randomRotation()],[array([0,0,8]),randomRotation()], [array([0,0,12]),randomRotation()]]
    #generate convolution
    positionsRMat=convolveBBCoords(BBTest[0],positions)
    positionsXYZ=convertSitesRMatToXYZ(positionsRMat)
    BBXYZ=convertSitesRMatToXYZ(BBTest)

    targetCoords=computeCoords(BBXYZ[0],positionsXYZ)
  
    printOutput('targetCoords',targetCoords,2)
    printOutput('positions',positions,2)
    sys.exit()

    return

def writeGeometry(BBGeometry,filename):
    f=open(filename,'w')
    angles=[ '('+str(pos[0][0])+' '+str(pos[0][1])+') '+  str(pos[1][0])+' '+str(pos[1][1])+' '+str(pos[1][2])+' '+str(pos[1][3])+' '+str(pos[1][4])+' '+str(pos[1][5])+'\n' for pos in BBGeometry]
    for pos in angles:
       f.write(pos)
    f.close()
    return


if __name__ == '__main__':
    rOn=1
    aOn=1

    #testWrap()
    #testAngle()
    #testRmatAAxis()
    #testPysites() 
    #testConvolve()
    #testGlobalRotate()
    #testPerformAlignment()
    #testAlign()
    #testAlignBB() 
    #failedAlign()
    #testComputeBB()
    #testComputeCoords()
    #This switch controls the output of the rmsd routine
    #0 = err for normal operation
    #1 = [rMatBest, alignDataXYZ]
    #2 = alignedDataXYZ
    #3 = output all data to file/stdout and stop on first run.
    output=0

    filename=getFilename()
    atoms=readPDB(filename+'.pdb')
    site=readSites(filename+'.sit')
    sitesRMat=analyseXYZ(site,atoms)
    sitesXYZ=convertSitesRMatToXYZ(sitesRMat)
    #sitesRMat2=convertSitesXYZToRMat(sitesXYZ)
    #writeXYZ(filename+'Analysis.RMat2.xyz',sitesRMat2)
    writeXYZ('Analysis.RMat.xyz',sitesRMat,0.0)
    writeXYZ('Analysis.XYZ.xyz',sitesXYZ,0.0)
    
    BBRMat=computeBB(sitesXYZ)
    writeXYZ('BB.Rmat.xyz',BBRMat,0.0)
    BBXYZ=convertSitesRMatToXYZ(BBRMat)
    writeXYZ('BB.Rmat.xyz',BBRMat,0.0)
    writeXYZ('BB.XYZ.xyz',BBXYZ,0.0)

    targetCoords=computeCoords(BBXYZ[0],sitesXYZ)
    sitesTargetRMat=convolveBBCoords(BBRMat[0],targetCoords)  
    sitesTargetXYZ=convolveBBCoords(BBXYZ[0],targetCoords)  
    writeXYZ('target.Rmat.xyz',sitesTargetRMat,0.0)
    writeXYZ('target.XYZ.xyz',sitesTargetXYZ,0.0)

    BBGeometry=analyseGeometry(BBRMat[0],filename+'.pair')
    print 'geometry:'
    print BBGeometry
    writeGeometry(BBGeometry,filename+'.angles')
    writeModel('model','coordsModel',BBRMat[0],BBGeometry[0][1],rOn,aOn,len(targetCoords))
    optimData=optimiseParams(BBRMat[0],targetCoords,rOn,aOn,output)
    BBRMatFinal=optimData[0]
    alignRMat=optimData[1][0]
    coordsFinal=optimData[1][1]
    writePYSites('pysites.xyz',BBRMatFinal,rOn,aOn)
    finalRMatSites=convolveBBCoords(BBRMatFinal,coordsFinal)
    writeXYZ('final.rmat.xyz',finalRMatSites,alignRMat[0])
    finalXYZSites=convertSitesRMatToXYZ(finalRMatSites)
    writeXYZ('final.xyz.xyz',finalXYZSites,alignRMat[0])

















 
 
          




 
