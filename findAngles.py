#!/usr/bin/python

import argparse
from numpy import *
from math import *

def readAtoms(filename):
    
    wantlist = ["ATOM","HETATM"]
    chucklist = ["ACE","NH2","HOH","ACY","AZI","CO3"]
    
    with open(filename, 'r') as f:
        
        goodlines = []
        
        for line in f.readlines():
            bits = line.split(None,15)
            if bits[0] in wantlist and bits[3] not in chucklist:
                for index in [1, 5]:
                    bits[index] =  int(bits[index])
                for index in range(6,11):
                    bits[index] = float(bits[index])
                goodlines.append(bits[1:])
    
    return goodlines

def findResidues(atoms):
    
    residueList=[]
    for atom in atoms:
        residueType = atom[2]
        residueNum = atom[4]
        residueChain = atom[3]
        residue = [residueType, residueNum, residueChain]
        
        if not residue in residueList:
            residueList.append(residue)

    return residueList

def calcTorsionangle(atom1,atom2,atom3,atom4):
    
    vec1 = atom2 - atom1
    vec2 = atom3 - atom2
    vec3 = atom4 - atom3
    calctangle = atan2( vdot( linalg.norm(vec2)*vec1 , cross(vec2,vec3) ), vdot( cross(vec1,vec2), cross(vec2, vec3)) )*180/pi
   
    return calctangle  # angle in degrees

def pickoneResidue(residueType, residueNum, residueChain):

    oneresidueatomsList = []
    for atom in atoms:
        if (atom[4]==residueNum) and (atom[3]==residueChain):
            oneresidueatomsList.append(atom)
        elif (atom[4]==residueNum-1) and (atom[3]==residueChain):
            oneresidueatomsList.append(atom)
        elif (atom[4]==residueNum+1) and (atom[3]==residueChain):
            oneresidueatomsList.append(atom)

    return oneresidueatomsList
    
def findTorsionAngle(residueType, residueNum, residueChain):
    #~ print "residueType = ",residueType
    atomworking = pickoneResidue(residueType, residueNum, residueChain)

    origin = array([0., 0., 0.])
    currCB, currCG, currCD, currNH1, currNH2 = [origin]*5
    prevC,nextN,nextCA = [origin]*3
    for atom in atomworking:
        atomtype = atom[1]
        atomresnum = atom[4]
        atomcoords = array([atom[5],atom[6],atom[7]])
        
        if (atomresnum==residueNum):
            
            currchain = atom[3]
                            
            if atomtype=='N':
                currN= atomcoords
                #~ print currN
            if atomtype=='CA':
                currCA= atomcoords
                #~ print currCA
            if atomtype=='C':
                #~ print atom
                currC= atomcoords
                #~ print currC
            if residueType=='PRO' or residueType=='HYP' or residueType=='MP8' or residueType=='FP9':
                if atomtype=='CB':
                    currCB= atomcoords
                if atomtype=='CG':
                    currCG= atomcoords
                if atomtype=='CD':
                    currCD= atomcoords
            if residueType=='PHE':
                if atomtype=='CB':
                    currCB= atomcoords
                if atomtype=='CG':
                    currCG= atomcoords
                if atomtype=='CD1':
                    currCD1= atomcoords
                if atomtype=='CD2':
                    currCD2= atomcoords
                if atomtype=='CE1':
                    currCE1= atomcoords
                if atomtype=='CE2':
                    currCE2= atomcoords
            if residueType=='GLU':
                if atomtype=='CB':
                    currCB= atomcoords
                if atomtype=='CG':
                    currCG= atomcoords
                if atomtype=='CD':
                    currCD= atomcoords
                if atomtype=='OE1':
                    currOE1= atomcoords
                if atomtype=='OE2':
                    currOE2= atomcoords
            if residueType=='ARG':
                if atomtype=='CB':
                    currCB= atomcoords
                if atomtype=='CG':
                    currCG= atomcoords
                if atomtype=='CD':
                    currCD= atomcoords
                if atomtype=='NE':
                    currNE= atomcoords
                if atomtype=='CZ':
                    currCZ= atomcoords
                if atomtype=='NH1':
                    currNH1= atomcoords
                if atomtype=='NH2':
                    currNH2= atomcoords
                    
        elif atomresnum==residueNum-1 and atomtype=='C':
            prevC= atomcoords

        elif atomresnum==residueNum+1:
            if atomtype=='N':
                nextN= atomcoords
            if atomtype=='CA':
                nextCA= atomcoords

    phi = calcTorsionangle(prevC,currN,currCA,currC)
    psi = calcTorsionangle(currN,currCA,currC,nextN)
    omega = calcTorsionangle(currCA,currC,nextN,nextCA)
    #print phi, psi, omega
    #~ #print prevC, currN, currCA, currC, nextN, nextCA
    #~ #print currCB, currCG, currCD
    #~ #print "phi", calcTorsionangle(prevC,currN,currCA,currC)
    #~ #print "psi", calcTorsionangle(currN,currCA,currC,nextN)
    #~ #print "omega", calcTorsionangle(currCA,currC,nextN,nextCA)
    #~ #if not array_equal(currCB,origin) or not array_equal(currCG,origin) or not array_equal(currCD,origin):
    if residueType=='PRO' or residueType=='HYP' or residueType=='MP8' or residueType=='FP9':
        chi_1 = calcTorsionangle(currN,currCA,currCB,currCG)
        chi_2 = calcTorsionangle(currCA,currCB,currCG,currCD)
        chi_3 = calcTorsionangle(currCB,currCG,currCD,currN)
        chi_4 = calcTorsionangle(currCG,currCD,currN,currCA)
        chi_5 = calcTorsionangle(currCD,currN,currCA,currCB)
        #print "chi_1", calcTorsionangle(currN,currCA,currCB,currCG)
        #print "chi_2", calcTorsionangle(currCA,currCB,currCG,currCD)
        #print "chi_3", calcTorsionangle(currCB,currCG,currCD,currN)
        #print "chi_4", calcTorsionangle(currCG,currCD,currN,currCA)
        #print "chi_5", calcTorsionangle(currCD,currN,currCA,currCB)
        print '%s-%s-%s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f' % (residueType, residueNum,currchain, phi, psi, omega, chi_1, chi_2, chi_3, chi_4, chi_5)
    elif residueType=='PHE':
        chi_1 = calcTorsionangle(currN,currCA,currCB,currCG)
        chi_2a = calcTorsionangle(currCA,currCB,currCG,currCD1)
        chi_3a = calcTorsionangle(currCB,currCG,currCD1,currCE1)
        print '%s-%s-%s %.3f %.3f %.3f %.3f %.3f %.3f' % (residueType, residueNum,currchain, phi, psi, omega, chi_1, chi_2a, chi_3a)
    elif residueType=='GLU':
        chi_1 = calcTorsionangle(currN,currCA,currCB,currCG)
        chi_2 = calcTorsionangle(currCA,currCB,currCG,currCD)
        chi_3a = calcTorsionangle(currCB,currCG,currCD,currOE1)
        print '%s-%s-%s %.3f %.3f %.3f %.3f %.3f %.3f' % (residueType, residueNum,currchain, phi, psi, omega, chi_1, chi_2, chi_3a)
    elif residueType=='ARG':
        chi_1 = calcTorsionangle(currN,currCA,currCB,currCG)
        chi_2 = calcTorsionangle(currCA,currCB,currCG,currCD)
        chi_3 = calcTorsionangle(currCB,currCG,currCD,currNE)
        chi_4 = calcTorsionangle(currCG,currCD,currNE,currCZ)
        chi_5a = calcTorsionangle(currCD,currNE,currCZ,currNH1)
        print '%s-%s-%s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f' % (residueType, residueNum,currchain, phi, psi, omega, chi_1, chi_2, chi_3, chi_4, chi_5a)
    else:
        print '%s-%s-%s %.3f %.3f %.3f' % (residueType, residueNum,currchain, phi, psi, omega)


if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(
        description = 'Python script to find all backbone (phi, psi, omega) dihedral angles and selected sidechain (chi) torsion angles from PDB file')
    parser.add_argument("PDBfile", help = "file for finding angles in PDB format")
    args = parser.parse_args()

    atoms = readAtoms(args.PDBfile)

    residueslist=findResidues(atoms)

    for eachresidue in residueslist:
        findTorsionAngle(*eachresidue)
