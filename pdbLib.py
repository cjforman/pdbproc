#!/usr/bin/env python
import sys, os
import numpy as np
import matplotlib.pyplot as plt  
from scipy.optimize import fmin_l_bfgs_b
from itertools import chain
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import copy as cp
import glob
import json
from setuptools.sandbox import AbstractSandbox


# A global dictionary containing the information on which side groups can rotate for the various amino acids.
# don't include the rotation axis in the list of atoms 
AARotTemplates = {
    
    'ACE': { ('C', 'CH3'): ['HH31', 'HH32', 'HH33'] },
    
    'ALA': { ('C', 'CA'): ['HA', 'CB', 'HB1', 'HB2', 'HB3'], 
             ('CA', 'N'): ['HA', 'CB', 'HB1', 'HB2', 'HB3'],
             ('CA', 'CB'): ['HB1', 'HB2', 'HB3'] },
    
    'ARG': { ('CA', 'N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'NE', 'HE', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'CZ'],
             ('CA', 'C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'NE', 'HE', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'CZ'],
             ('CA', 'CB'): ['HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'NE', 'HE', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'CZ'],
             ('CB', 'CG'): ['HG2', 'HG3', 'CD', 'HD2', 'HD3', 'NE', 'HE', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'CZ'],
             ('CG', 'CD'): ['HD2', 'HD3', 'NE', 'HE', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'CZ'],
             ('CD', 'NE'): ['HE', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'CZ'],
             ('NE', 'CZ'): ['NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22'],
             ('CZ', 'NH1'): ['HH11', 'HH12'],
             ('CZ', 'NH2'): ['HH21', 'HH22']},
    
    'ASN': { ('CA', 'N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'ND2', 'HD21', 'HD22', 'OD1'],
             ('CA', 'C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'ND2', 'HD21', 'HD22', 'OD1'],
             ('CA', 'CB'): ['HB2', 'HB3', 'CG', 'ND2', 'HD21', 'HD22', 'OD1'],
             ('CB', 'CG'): ['ND2', 'HD21', 'HD22', 'OD1'],
             ('CG', 'ND2'): ['HD21', 'HD22']},
    
    'ASP': { ('CA', 'C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'OD2'],
             ('CA', 'N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'OD1', 'OD2'],
             ('CA','CB'): ['HB2', 'HB3', 'CG', 'OD1', 'OD2'],
             ('CB','CG'): ['OD1', 'OD2'] },
    
    'CYS': { ('CA','C'): ['HA', 'CB', 'HB2', 'HB3', 'SG', 'HG'],
             ('CA','N'): ['HA', 'CB', 'HB2', 'HB3', 'SG', 'HG'],
             ('CA','CB'): ['HB2', 'HB3', 'SG', 'HG'], 
             ('CB','SG'): ['HG'] },
    
    'GLU': { ('CA','N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'OE2'],
             ('CA','C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'OE2'],
             ('CA','CB'): ['HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'OE2'],
             ('CB','CG'): ['HG2', 'HG3', 'CD', 'OE1', 'OE2'],
             ('CG','CD'): ['OE1', 'OE2'] },

    'GLY': { ('CA','N'): ['HA2', 'HA3' ],
             ('CA','C'): ['HA2', 'HA3' ] },
    
    'GLN': { ('CA','C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'NE2', 'HE21', 'HE22'],
             ('CA','N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'NE2', 'HE21', 'HE22'],
             ('CA','CB'): ['HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'OE1', 'NE2', 'HE21', 'HE22'],
             ('CB','CG'): ['HG2', 'HG3', 'CD', 'OE1', 'NE2', 'HE21', 'HE22'],
             ('CG','CD'): ['OE1', 'NE2', 'HE21', 'HE22'],
             ('CD','NE2'): ['HE21', 'HE22']},
    
    'HIS': { ('CA','N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'CD2', 'HD2', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2'],
             ('CA','C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'CD2', 'HD2', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2'],
             ('CA','CB'): ['HB2', 'HB3', 'CG', 'CD2', 'HD2', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2'],
             ('CB','CG'): ['CD2', 'HD2', 'ND1', 'HD1', 'CE1', 'HE1', 'NE2'],
             ('ND1','NE2'): ['HD1', 'CE1', 'HE1']},
    
    'HYP': { ('CA','N'): ['CB', 'HB2', 'HB3', 'CG', 'HG2', 'OD1', 'HO1', 'CD', 'HD2', 'HD2', ],
             ('CB','CD'): ['HB2', 'HB3', 'CG', 'HG2', 'OD1', 'HO1', 'HD2', 'HD2', ],
             ('CG','OD'): ['HO1']},
    
    'ILE': { ('CA','N'): ['HA', 'CB', 'HB', 'CG1', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23', 'CD1', 'HD11', 'HD12', 'HD13'],
             ('CA','C'): ['HA', 'CB', 'HB', 'CG1', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23', 'CD1', 'HD11', 'HD12', 'HD13'],
             ('CA','CB'): ['HB', 'CG1', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23', 'CD1', 'HD11', 'HD12', 'HD13'],
             ('CB','CG1'): ['HG12', 'HG13', 'CD1', 'HD11', 'HD12', 'HD13'],
             ('CG1','CD1'): [ 'HD11', 'HD12', 'HD13'],
             ('CB','CG2'): [ 'HG21', 'HG22', 'HG23']},
    
    'LEU': { ('CA','N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22', 'HD23'],
             ('CA','C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22', 'HD23'],
             ('CA','CB'): ['HB2', 'HB3', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22', 'HD23'], 
             ('CB','CG'): ['HG', 'CD1', 'HD11', 'HD12', 'HD13', 'CD2', 'HD21', 'HD22', 'HD23'],
             ('CG','CD1'): ['HD11', 'HD12', 'HD13'],
             ('CG','CD2'): ['HD21', 'HD22', 'HD23']},
    
    'LYS': { ('CA','C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ', 'HZ1', 'HZ2', 'HZ3'],
             ('CA','N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ', 'HZ1', 'HZ2', 'HZ3'],
             ('CA','CB'): ['HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ', 'HZ1', 'HZ2', 'HZ3'],
             ('CB','CG'): ['HG2', 'HG3', 'CD', 'HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ', 'HZ1', 'HZ2', 'HZ3'],
             ('CG','CD'): ['HD2', 'HD3', 'CE', 'HE2', 'HE3', 'NZ', 'HZ1', 'HZ2', 'HZ3'],
             ('CD','CE'): ['HE2', 'HE3', 'NZ', 'HZ1', 'HZ2', 'HZ3'],
             ('CE','NZ'): ['HZ1', 'HZ2', 'HZ3']},
    
    'MET': { ('CA','C'): ['CB', 'HA', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'SD', 'CE', 'HE2', 'HE3'],
             ('CA','N'): ['CB', 'HA', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'SD', 'CE', 'HE2', 'HE3'],
             ('CA','CB'): ['HB2', 'HB3', 'CG', 'HG2', 'HG3', 'SD', 'CE', 'HE2', 'HE3'],
             ('CB','CG'): ['HG2', 'HG3', 'SD', 'CE', 'HE2', 'HE3'],
             ('CG','SD'): ['CE', 'HE2', 'HE3'],
             ('SD','CE'): ['HE2', 'HE3'] },

    'NME': { ('N', 'CH3'): ['HH31', 'HH32', 'HH33'] },

    'PHE': {  ('CA','C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'HD2', 'CE1', 'HE1', 'CE2', 'HE2', 'CZ', 'HZ1'],
              ('CA','N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'HD2', 'CE1', 'HE1', 'CE2', 'HE2', 'CZ', 'HZ1'],
              ('CA','CB'): ['HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'HD2', 'CE1', 'HE1', 'CE2', 'HE2', 'CZ', 'HZ1'], 
              ('CB','CG'): ['CD1', 'HD1', 'CD2', 'HD2', 'CE1', 'HE1', 'CE2', 'HE2', 'CZ', 'HZ1']},
    
    'PRO': { ('CA','N'): [ 'HA', 'CB', 'HB2', 'HB3', 'CG', 'HG2', 'HG3', 'CD', 'HD2', 'HD2'],
             ('CB','CD'): ['HB2', 'HB3', 'CG', 'HG2', 'HG3', 'HD2', 'HD2'] },
    
    'SER': { ('CA', 'C'): ['HA', 'CB', 'HB2', 'HB3', 'OG', 'HG'],
             ('CA', 'N'): ['HA', 'CB', 'HB2', 'HB3', 'OG', 'HG'],
             ('CA', 'CB'): ['HB2', 'HB3', 'OG', 'HG'],
             ('CB', 'OG'): ['HG'] },

    'THR': { ('CA', 'N'): ['HA', 'CB', 'HB', 'OG1', 'HG1', 'CG2', 'HG21', 'HG22', 'HG23'],
             ('CA', 'C'): ['HA', 'CB', 'HB', 'OG1', 'HG1', 'CG2', 'HG21', 'HG22', 'HG23'],
             ('CA', 'CB'): ['HB', 'OG1', 'HG1', 'CG2', 'HG21', 'HG22', 'HG23'], 
             ('CB', 'OG1'): ['HG1'],
             ('CB', 'CG2'): ['HG21', 'HG22', 'HG23']},
    
    'TRP': { ('CA', 'C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'NE1', 'HE1', 'CE2', 'CE3', 'HE3', 'CZ2', 'HZ2', 'CZ3', 'HZ3', 'CH2', 'HH2'],
             ('CA', 'N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'NE1', 'HE1', 'CE2', 'CE3', 'HE3', 'CZ2', 'HZ2', 'CZ3', 'HZ3', 'CH2', 'HH2'],
             ('CA', 'CB'): ['HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'NE1', 'HE1', 'CE2', 'CE3', 'HE3', 'CZ2', 'HZ2', 'CZ3', 'HZ3', 'CH2', 'HH2'],
             ('CB', 'CG'): ['CD1', 'HD1', 'CD2', 'NE1', 'HE1', 'CE2', 'CE3', 'HE3', 'CZ2', 'HZ2', 'CZ3', 'HZ3', 'CH2', 'HH2'] },

    'TYR': { ('CA', 'C'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'HD2', 'CE1', 'HE1', 'CE2', 'HE2', 'CZ', 'OH', 'HH'],
             ('CA', 'N'): ['HA', 'CB', 'HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'HD2', 'CE1', 'HE1', 'CE2', 'HE2', 'CZ', 'OH', 'HH'],
             ('CA', 'CB'): ['HB2', 'HB3', 'CG', 'CD1', 'HD1', 'CD2', 'HD2', 'CE1', 'HE1', 'CE2', 'HE2', 'CZ', 'OH', 'HH'],
             ('CB', 'CG'): ['CD1', 'HD1', 'CD2', 'HD2', 'CE1', 'HE1', 'CE2', 'HE2', 'CZ', 'OH', 'HH'],
             ('CZ', 'OH'): ['HH']},
    
    'VAL': { ('CA', 'C'): ['HA', 'CB', 'HB', 'CG1', 'HG11', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23'],
             ('CA', 'N'): ['HA', 'CB', 'HB', 'CG1', 'HG11', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23'],
             ('CA', 'CB'): ['HB', 'CG1', 'HG11', 'HG12', 'HG13', 'CG2', 'HG21', 'HG22', 'HG23'],
             ('CB', 'CG1'): [ 'HG11', 'HG12', 'HG13'],
             ('CB', 'CG2'): [ 'HG21', 'HG22', 'HG23']}
    }

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def generateRigidBodyInput(infile, Cis_Trans_File):

    # load the pdb file and dump as a coordsinirigid file
    atoms = readAllAtoms(infile)
    outlist= []
    for atom in atoms:
        outlist.append(str(atom[7]) + " " + str(atom[8]) + " " + str(atom[9]) + "\n")
        writeTextFile(outlist, "coordsinirigid")

    outlist=[]
    # use the cis trans infor to generate a set of rigid body groups
    peptideBondGroupList = [ [int(val) for val in PBG.strip('\r\n').split()] for PBG in readTextFile(Cis_Trans_File)[0::2] ]

    for rigidBodyGroup in peptideBondGroupList:
        outlist.append( "GROUP " + str( len( rigidBodyGroup ) ) + "\n" )
    for atomIndex in rigidBodyGroup:
        outlist.append( str( atomIndex ) + "\n" )
    outlist.append("\n")

    writeTextFile(outlist, "rbodyconfig")


# function generates rigidbody coords and atomgroups for a freezedry simulation in which 
# residues 1 to rBodyId1 is a rigid body
# residues rBodyId2 to end is a rigid body
# Each N and CA atom in residues rBodyId1 + 1 to rBodyId2 - 1 define an atom rotation group 
# about the N-CA axis from N to the end of the chain including sidechain of residue containing N and CA   
# about CA-C axis from CA to the end of the chain excluding the sidechain of the residue containing CA and C. 
def generateFreezeDryGroups(infile, rBody1Id, rBody2Id, angRange, includeSideChains=False):

    # get all the atoms
    atoms = readAllAtoms(infile)

    # re-assign the chain letters to each atom so there are distinct chain letters for each chain 
    assignChains(atoms)
    
    # generate list of atoms in each chain - for the rest assume there is only one chain. Gonna get messy otherwise.  
    chains = breakAtomsInToChains(atoms)
    
    if len(chains)!=1:
        print("Warning: attempting freezedry routine on pdb with more than one chain. Results are gonna be a mess.")
    
    # for each chain break the list into residue sets
    ChainResidues = [ breakAtomsIntoResidueDicts(chain) for chain in chains ]
    
    # count the overall number of residues to give each "residue to end" group a 1/2N chance of being picked. 
    # 2 groups come out of each residue.
    numResidues = rBody2Id - rBody1Id + 1
    probSelect = 1.0/(2.0 * float(numResidues))
    
    # create a list of dictionaries that defines each rotation group  
    atomGroupsDictList = []
    
    # loop through each chain - algorithm will work with more than 1 chain but results are gonna be a mess with more than 1 chain. 
    for chain, residues in zip(chains, ChainResidues):
        
        # loop through each residue in the chain and extract the list of atoms from the dictionary 
        for _, residue in residues.items():
            resid = residue[0][5]
            
            # Include residues whose ids lie between the specified values (including the end points)
            if resid>=rBody1Id and resid<=rBody2Id:

                for atom in residue:
                    if atom[1]=='C':
                        C = cp.copy(atom)
                    if atom[1]=='CA':
                        CA = cp.copy(atom)
                    if atom[1]=='N':
                        N = cp.copy(atom)
                        
                # add a dictionary to the list which contains an atomgroup consisting of an axis N-CA in current residue, 
                # and atoms from the N to the end of each chain
                atomGroupsDictList.append( createEndOfChainRotationGroupDict2(chain, N, CA, probSelect, rotScaleFactor=angRange) )
                
                # add a dictionary to the list which contains an atomgroup consisting of an axis CA-C in current residue, 
                # and atoms from the N to the end of each chain
                atomGroupsDictList.append( createEndOfChainRotationGroupDict2(chain, CA, C, probSelect, rotScaleFactor=angRange) )

    if includeSideChains:
        print("Adding Side Chain Rotations") # for those groups within the range including end points of the range.
        atomGroupsDictList += makeSideChainsRotationGroupDictList(atoms, minRes=rBody1Id, maxRes=rBody2Id)
    else:
        print("No Side Chain Rotations were added")

    # write the atomgroups file
    print("Writing Atomgroups file")
    writeAtomGroups("atomgroups", atomGroupsDictList)
    
    
    # write out the atom groups as separate PDBS for debugging. 
    # for rotGroupDict in atomGroupsDictList:
    #     writeAtomsToTextFile(rotGroupDict['atoms'], 'groupPDBS/'+rotGroupDict['name'] + '.pdb')

    # now do the rigidbodies - write the coordsini rigid file
    configOutlist = []
    for atom in atoms:
        configOutlist.append(str(atom[7]) + " " + str(atom[8]) + " " + str(atom[9]) + "\n")
        writeTextFile(configOutlist, "coordsinirigid")

    rigidBodyList1 =[]
    rigidBodyList2 =[]
    
    outList = []
    for atom in atoms:
        if atom[5]<=rBody1Id:
            rigidBodyList1.append(atom[0])
    
        if atom[5]>=rBody2Id:
            rigidBodyList2.append(atom[0])

    for rBGroup in [rigidBodyList1, rigidBodyList2]:
        outList.append( "GROUP " + str( len( rBGroup ) ) + "\n" )
        for atomIndex in rBGroup:
            outList.append( str( atomIndex ) + "\n" )
        outList.append("\n")

    print("Writing Rigidbody file")
    writeTextFile(outList, "rbodyconfig")


    print("Done")


# checks a cis_trans_state file for CIS atoms. 
# Returns a list of GMIN atomic indices of atoms involved in the errant dihedral if there are any.
def checkFileForCIS(filename):
    
    # load the data
    data = readTextFile(filename)
    
    # parse the atom and CTStates from the data
    atoms = [ line.strip('\r\n') for line in data[0::2] ] 
    CTStates = [ line.strip('\r\n') for line in data[1::2] ]

    # generate output list
    outList = []
    
    # loop through the data
    for atom, CTState in zip(atoms, CTStates):
        # if the state is a CIS state add the atoms in the group to out list as indexes
        if 'C' in CTState:
            # atoms vals are the O C N H index in the PDB of the peptide bond always.
            outList.append([ int(index) for index in atom.split()])

    return outList

    
# splits the peptide bond into two groups and rotates them random amounts each.
# leads to transitions between CIS and Trans isomerisations.    
def makeAtomGroupsFile(atomList, atoms):
    outStrings = []
    for atomGroup in atomList:
        # atomGroup are the O C N H index in the PDB of the peptide bond always.
        # Getthe index of the CA in same residue as the Oxygen (atomGroup[0])
        resValOfTheO = atoms[atomGroup[0] - 1][5]
        CAIndex = findIndexVal(atoms, 'CA', resValOfTheO)
        
        # Create atom group to perform first rotation of the O and C about the N CA axis:
        name = str(resValOfTheO) + "_" + str(atomGroup[0]) + "_1" 
        outStrings.append( "GROUP " + name + " " + str(atomGroup[2]) + " " + str(CAIndex) + " 2 0.5 1.0\n" )
        outStrings.append( str(atomGroup[0]) + "\n" )
        outStrings.append( str(atomGroup[1]) + "\n\n" )
        
        # Getthe index of the CA in same residue as the Nitrogen (atomVals[2])
        resValOfTheN = atoms[atomGroup[2] - 1][5]
        CAIndex = findIndexVal(atoms, 'CA', resValOfTheN)
        
        # create the atom group for the second rotation of the N and H about the C CA axis:
        name = str(resValOfTheO) + "_" + str(atomGroup[0]) + "_2"
        outStrings.append( "GROUP " + name + " " + str(atomGroup[1]) + " " + str(CAIndex) + " 2 0.5 1.0\n" )
        outStrings.append( str(atomGroup[2]) + "\n" )
        outStrings.append( str(atomGroup[3]) + "\n\n" )
        
    writeTextFile(outStrings, "atomgroups")

def backboneAtomGroups(infile, angRange):

    # get all the atoms
    atoms = readAllAtoms(infile)
    
    # re-assign the chain letters to each atom so there are distinct chain letters for each chain 
    assignChains(atoms)
    
    # generate list of atoms in each chain
    chains = breakAtomsInToChains(atoms)
    
    # for each chain break the list into residue sets
    ChainResidues = [ breakAtomsIntoResidueDicts(chain) for chain in chains ]
    
    # count the overall number of residues to give each group 1/2N chance of being picked. (2 groups come out of each residue)
    numResidues = sum([len(chain) for chain in ChainResidues])
    probSelect = 1.0/(2.0 * float(numResidues))
    
    # create a list of dictionaries that defines each rotation group 
    atomGroupsDictList = [] 
    
    # loop through each chain 
    for chain, residues in zip(chains, ChainResidues):
        
        # loop through each residue in the chain and extract the list of atoms from the dictionary 
        for _, residue in residues.items():

            # for each residue create a dictionary with a list of all atoms from CA in that residue to the end of each chain
            atomGroupsDictList.append( createEndOfChainRotationGroupDict(chain, residue, 'CA', probSelect, rotScaleFactor=angRange) )
            
            # for each residue create a dictionary with a list of all atoms from N in that residue to the end of each chain
            atomGroupsDictList.append( createEndOfChainRotationGroupDict(chain, residue, 'N', probSelect, rotScaleFactor=angRange) )

    # write the atomgroups file
    print("Writing Atomgroups file")
    writeAtomGroups("atomgroups", atomGroupsDictList)
    # write out the atom groups as separate PDBS for debugging. 
    # for rotGroupDict in atomGroupsDictList:
    #     writeAtomsToTextFile(rotGroupDict['atoms'], 'groupPDBS/'+rotGroupDict['name'] + '.pdb')

    print("Done")




# generates an atom groups file that contains rotation groups defined by the 
# segment of a chain between CAs that obey the restrictions.
# Can include the side chains on the limiting CAs or not as desired
# For good measure also can add all side chain rotations to the atomgroups file if desired 
def crankShaftGroups(infile, nnCutoff, dCutoff, includeSideChains=False, rotateSideGroups=False, rotScale=0.1):
    # get all the atoms
    atoms = readAllAtoms(infile)
    
    # re-assign the chain letters to each atom so there are distinct chain letters for each chain 
    assignChains(atoms)
    
    # extract all the CAs in the entire protein into a sublist
    CAList = [atom for atom in atoms if atom[1]=='CA']
    
    # set up a dict keyed by each CA to store which other CAs may form end points with each key.  
    CADict = {}
    
    # for each CA identify the list of CAs that will form end points of crankChains with this CA.
    for CAListIndex, CA in enumerate(CAList):
        # CAtest qualifies as being an end point with CA if the following apply:
        # CA and CATtest are in the same chain
        # CATest's residue number is within nnCutoff residues of the key CA
        # If CAtest is within dCutoff euclidean distance
        # CA and CATest are not the same atom 
        # the CA's are noted by their index number in the CAList for efficient look up
        CADict[CAListIndex] = [ CAListIndexTest for CAListIndexTest, CAtest in enumerate(CAList) if testCAPair(CA, CAtest, nnCutoff, dCutoff)==True ]

    # from the dictionary generate a set of pairs of CAListIndices ordered lowest first. 
    CAPairs = []
    for CAListIndex1 in CADict:
        for CAListIndex2 in CADict[CAListIndex1]:
            if CAListIndex1 < CAListIndex2:
                CAPairs.append( (CAListIndex1, CAListIndex2) )
            else:
                CAPairs.append( (CAListIndex2, CAListIndex1) ) 
     
    # use the set functionality to eliminate duplicates
    CAPairs = set(CAPairs)

    if len(CAPairs)==0:
        print("No CA Pairs were selected.")
        probSelect = 0.0
    else:
        # set ProbSelect to be 1/number of crank chains
        probSelect = 1.0/float(len(CAPairs))
   
    # a list of dictionaries that define the rotation groups 
    atomGroupsDictList = [] 
    
    # for each pair generate a list of atoms that will go in the crank chain.
    # add that list of atoms as a sublist to atomGroupsList
    # de-index the CAList so the inputs to the sub routine are the actual CA atoms properties list
    for CAPair in CAPairs:
        # generate the rotation group dictionary and add it
        atomGroupsDictList.append(createRotationGroupDict(atoms, CAList[CAPair[0]], CAList[CAPair[1]], probSelect, includeSideChains=includeSideChains, rotScaleFactor=rotScale))

    # if the rotate side group flag is set, then add all rotatable groups in sidechains in the protein to the list.
    if rotateSideGroups:
        print("Adding Side Chain Rotations")
        atomGroupsDictList += makeSideChainsRotationGroupDictList(atoms)
    else:
        print("No Side Chain Rotations were added")

    # write the atomgroups file
    print("Writing Atomgroups file")
    writeAtomGroups("atomgroups", atomGroupsDictList)
    # write out the atom groups as separate PDBS for debugging. 
    for rotGroupDict in atomGroupsDictList:
        writeAtomsToTextFile(rotGroupDict['atoms'], 'groupPDBS/'+rotGroupDict['name'] + '.pdb')


    print("Done")
    
# function to determine if a CA pair should be retained or not
def testCAPair(CA1, CA2, nnCutoff, dCutoff):
    goodPair = False
    if CA1[4]==CA2[4]:
        if ( np.abs(CA1[5] - CA2[5]) < nnCutoff ):
            if ( dist(CA1, CA2)<dCutoff):
                if ( CA1[0] != CA2[0] ):
                    goodPair=True
    return goodPair


# takes a list of dictionaries each of which defines a rotation group and outputs them all to filename
# in the atomgroups file format. 
def writeAtomGroups(filename, atomGroupDictList):
    
    # create output string
    outStrings = []
    
    try:
        # for each group extract the information from the dictionary and create the group definition line
        for rotGroupDict in atomGroupDictList:

            # only dump the group to file if there are atoms in the set            
            if len(rotGroupDict['atoms'])>0:
                outStrings.append( "GROUP " + rotGroupDict['name'] + " " + 
                                   str(rotGroupDict['start'][0]) + " " + 
                                   str(rotGroupDict['end'][0]) + " " +
                                   str(rotGroupDict['numAtoms']) + " " +
                                   str(rotGroupDict['rotScaleFactor']) + " " +
                                   str(rotGroupDict['probSelect']) + "\n" )
        
                # output the index number of each atom in the group on its own line (might have to add one to this (and axis atoms)        
                for atom in rotGroupDict['atoms']:
                    outStrings.append( str(atom[0]) + "\n" ) 
        
                # add a few line feeds to space the groups out        
                outStrings.append( "\n\n" )
    except KeyError as e:
        print("Error: group rotation information missing", e)
        sys.exit()
    
    writeTextFile(outStrings, filename)

#     0          1          2           3          4          5         6           7         8          9
# atom_seri, atom_name, alte_loca, resi_name, chai_iden, resi_numb, code_inse, atom_xcoo, atom_ycoo, atom_zcoo, atom_occu, atom_bfac,seg_id,atom_symb,charge
    

# loops through the residues defined in the atoms list and 
# generates a list of dictionaries each defining possible side chain rotations for each kind of residue 
def makeSideChainsRotationGroupDictList(atoms, minRes=0, maxRes=np.inf):
    # creates a dictionary of lists of atoms in each residue keyed by the residue numbers
    residues = breakAtomsIntoResidueDicts(atoms)
    
    # computes probability of selecting each residue as 1/number of residues
    probSelect = 1.0 / float(maxRes - minRes + 1)

    # create an empty list for output 
    outList = []
    
    # loop through each residue in the residues dictionary
    for residue in residues:
        # only use residues within a certain range. Defaults from 0 to infinity
        if residue>=minRes and residue<=maxRes: 
            # takes the list of atoms define in each residue dictionary entry and returns a list of rotation group dictionaries,
            # there is one such rotation group dictionary for each rotatable sub group of the side chain.
            # sets the probability of selection of each group to be 1/number of residues
            outList += createSideChainRotationGroupDictList(residues[residue], probSelect=probSelect)

    return outList

# returns a dictionary containing lists of atoms in each residue keyed by the residue number
# the dictionary is ordered by the key
def breakAtomsIntoResidueDicts(atoms):
    outDict = {}
    for atom in atoms:
        try:
            outDict[atom[5]].append(atom)
        except KeyError:
            outDict[atom[5]] = [ atom ]
    
    # return the dictionary in residue order
    return {key:outDict[key] for key in sorted(outDict)}

# returns a list of dictionaries of all the rotatable side groups in a residue.
# dictionaries in the output list are of the form: {'numAtoms':0, 'atoms':[],'start':atom1, 'end':atom2}
# input is a list of atoms in the residue    
def createSideChainRotationGroupDictList(resAtomList, probSelect=0.1):

    # set up output list
    outDictList = []
    
    # identify the unique residue names in the residue atom list  
    resSet = set([ atom[3] for atom in resAtomList ])
    
    
    # res set should only have a single element which is the residue type of the residue
    if len(resSet)==1:
        resName = next(iter(resSet))
        resNum = resAtomList[0][5]
        
        # loop through the rotPair, atomsRotGroup pairing from the appropriate residue template from the template dictionary 
        for rotPair, atomRotGroup in AARotTemplates[resName].items():
            # rotPair defines the axis atom names 
            # AtomRotGroup defines the set of atoms to move
            # generate a dictionary containing all the necessary information concerning which atoms are in the rotation 
            # group in same format as other functions and add to the list. 
            outDictList.append({ 'numAtoms': len(atomRotGroup), 
                                 'atoms': [ atom for atom in resAtomList if atom[1] in atomRotGroup ],
                                 'start': [ atom for atom in resAtomList if atom[1]==rotPair[0] ][0],
                                 'end': [ atom for atom in resAtomList if atom[1]==rotPair[1] ][0],
                                 'name': resName + "_" + str(resNum) + "_" + rotPair[0] + "_" + rotPair[1],
                                 'rotScaleFactor': 1.0,
                                 'probSelect':  probSelect } )
    else:
        print("Error: residue atom list contains atoms from another residue.")
    
    return outDictList


# creates a dictionary storing all the atoms from the given atom in the given residue until the end of the chain.
# leaves out the side chain from the current residue.
def createEndOfChainRotationGroupDict(atoms, residueAtomList, atomName, probSelect, rotScaleFactor=0.1, endAxisAtom='last'):
    startAtom = None
    outDict = {'numAtoms':0, 'atoms':[]} #returns an empty dictionary of a start atom cannot be found. 
     
    for atom in residueAtomList:
        if atom[1]==atomName:
            startAtom = atom
    
    if startAtom:
        if endAxisAtom=='last':
            endAtom = atoms[-1]
        else:
            endAtom = endAxisAtom
            
                
        # set up a dictionary for the chain from named atom in residue to the last atom in the chain.
        outDict = {'numAtoms':0, 
                   'atoms':[],
                   'start':startAtom, 
                   'end':endAtom, 
                   'name': startAtom[1] + "_" + str(startAtom[5]), 
                   'rotScaleFactor': rotScaleFactor,
                   'probSelect': probSelect }
        
        # assert the start and end atoms are in the same chain and start atom is either an 'N' or a 'CA'
        if startAtom[4]==atoms[-1][4] and startAtom[1] in ['CA', 'N']:
            # create a list of atoms that meet the requirements for being in the current rotation group 
            outDict['atoms'] = [ atom for atom in atoms if atomInRotationGroupType2(atom, startAtom) ]
            # count the number of atoms in the rotation group.
            outDict['numAtoms'] = len(outDict['atoms'])
    
    return outDict

# creates a dictionary to store information about a rotation group. 
# the axis is defined by atom1 and atom2 (which is either a N-CA or CA-C pair in same residue. 
# The rotation group contains all the atoms from atom 1 until the end of the chain. 
# If atom1 is CA It does not include the side chain of the residue containing atom1. 
def createEndOfChainRotationGroupDict2(atoms, atom1, atom2, probSelect, rotScaleFactor=0.1):
    
    outDict = {'numAtoms':0, 'atoms':[]} # returns an empty dictionary. Won't result in a group in atom group file  
     
    # assert the start and end atoms are in the same chain 
    # and either atom1 is 'N' and atom2 is CA
    #          or atom1 is 'CA' and atom2 is C
    if atom1[4]==atoms[-1][4] and ( ( atom1[1]=='N' and atom2[1]=='CA' ) or ( atom1[1]=='CA' and atom2[1]=='C') ):
    
        # set up a dictionary for the group 
        outDict = {'numAtoms':0, 
                   'atoms':[],
                   'start':atom1, 
                   'end':atom2, 
                   'name': atom1[1] + "_" + str(atom1[5]), 
                   'rotScaleFactor': rotScaleFactor,
                   'probSelect': probSelect }
        
        # create a list of atoms that meet the requirements for being in the current rotation group 
        outDict['atoms'] = [ atom for atom in atoms if atomInRotationGroupType2(atom, atom1) ]
        
        # count the number of atoms in the rotation group.
        outDict['numAtoms'] = len(outDict['atoms'])
    
    return outDict



# creates a dictionary for a crank chain     
def createRotationGroupDict(atoms, atom1, atom2, probSelect, includeSideChains=True, rotScaleFactor=0.1):
    # set up a dictionary for the chain between atom1 and atom2.
    outDict = {'numAtoms':0, 
               'atoms':[],
               'start':atom1, 
               'end':atom2, 
               'name': atom1[3] + "_" + str(atom1[5]) + "_" + atom2[3] + "_" + str(atom2[5]), 
               'rotScaleFactor': rotScaleFactor,
               'probSelect': probSelect }
    
    # assert the atoms are in the same chain and are CA atoms. If not returns an empty chain, 
    # which won't be added to the atom groups file 
    if atom1[4]==atom2[4] and atom1[1]=='CA' and atom2[1]=='CA':
        # create a list of atoms that meet the requirements for being in the current rotation group 
        outDict['atoms'] = [ atom for atom in atoms if atomInRotationGroup(atom, atom1, atom2, includeSideChains=includeSideChains) ]
        # count the number of atoms in the rotation group.
        outDict['numAtoms'] = len(outDict['atoms'])
    
    return outDict

# atomgroup is selected by choosing every atom from the startatom residue CA or N to the end of the chain.
# if startAtom is CA omit the side chain for that residue. 
# if startAtom is N include the side chain for that residue.
# returns true if the test atom is in the atomgroup defined by the startAtom.
# returns true if test atom is:  
# 1) in a residue after the residue containing the startAtom
# 2) in the same residue at the start atom and:
#    1) If the startAtom is a CA, and the test atom is a 'C' or 'O' atom
#    2) If the startAtom is a N, and the test atom is not 'N' or 'H' atom. 
def atomInRotationGroupType2(testAtom, startAtom):
    # assume the atom is not in the group
    atomInGroup = False
    
    # if the atom is in a residue after the test residue then include it.  
    if testAtom[5]> startAtom[5]:
        atomInGroup = True

    # if the test atom is in the startAtom residue        
    elif testAtom[5]==startAtom[5]:
        # if the startAtom is a 'CA'
        if startAtom[1]=='CA':
            # if the test atom is one of C or O, add it to the group. 
            if testAtom[1] in ['C', 'O']:
                atomInGroup = True
        # if the startAtom is an 'N'
        if startAtom[1]=='N':
            # if the test atom is not N or H add it to the group. 
            if not testAtom[1] in ['N', 'H']: 
                atomInGroup = True
    
    return atomInGroup


# returns true if the test atom meets the requirements for being in a rotation group defined by atom1 and atom2
# assumes atom1 and atom2 are CA atoms. Atom1 and Atom2 are always added to the group. 
def atomInRotationGroup(atom, atom1, atom2, includeSideChains=True):
    # assume the atom is not in the group
    atomInGroup = False
    
    # if the atom is in an intervening residue the atom is in the group.
    if atom[5]> atom1[5] and atom[5]< atom2[5]:
        atomInGroup = True

    # if the atom is in the first residue        
    elif atom[5]==atom1[5]:
        # if the atom is one of C or O, add it to the group. 
        if atom[1] in ['C', 'O']:
            atomInGroup = True

        # if the side chains are being included, then only include the atom if it is not N or H. 
        if includeSideChains and not atom[1] in ['CA', 'N', 'H']:
            atomInGroup = True

    # if the atom is in the last residue        
    elif atom[5]==atom2[5]:
        # if the atom is one of N or H, add it to the group. 
        if atom[1] in ['N', 'H']:
            atomInGroup = True

        # if the side chains are being included, then only include the atom is it is not C or O. 
        if includeSideChains and not atom[1] in ['CA', 'C', 'O']:
            atomInGroup = True

    return atomInGroup

def dist(atom1, atom2):
    d = np.linalg.norm( np.array( [ atom1[7], atom1[8], atom1[9]] ) - np.array( [atom2[7], atom2[8], atom2[9]] ) )
    return d 
    
def assignChains(atoms):
    # reassigns the chain letters in the atom list with two
    # conditions for finding the end of the chain:
    # 1) the presence of an NME indicates to increment the chain
    # 2) the existing current chain letter is different that the last chain letter. 
    
    # initialize variables:
    chainLets=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    chainIndex = 0
    firstResidue = atoms[0][5] # residue number
    lastChainLetter = atoms[0][4] # chain identity
    prevResidue = atoms[0][3]
    
    # loop the atoms
    for atom in atoms:

        # if we hit an ACE when prev residue was an NME then increment the chain index, unless it's the first ACE. 
        if atom[3]=='ACE' and prevResidue=='NME' and atom[5]>firstResidue:
            chainIndex += 1
        
        # if the current chain letter is different than the last chain Letter than increment chain Index
        elif atom[4]!=lastChainLetter:
            chainIndex += 1
            lastChainLetter = atom[4]

        # write out the chain letter for this atom. Might relabel the chain letters from the order they are in.             
        atom[4] = chainLets[chainIndex]

        prevResidue = atom[3]
    
def findBestCoords(globMask):
    # get a glob of all pdb files
    pdbFiles = glob.glob(globMask)

    # create a CT file for every coords.x.pdb and record the number of CIS states in each PDB in a list. 
    # use the initial_cis_trans_states file as a template for peptide bonds and a threshold of 30 degrees to 
    # define what constitutes a window around a cis bond
    numCStates = [ createCTFile(filename, ["initial_cis_trans_states", 30.0]) for filename in pdbFiles ]

    minCStateIndex = numCStates.index(min(numCStates))

    return numCStates[minCStateIndex], pdbFiles[minCStateIndex]


def loadChiralCentres(chiralFile):
    # parses the chirality file
    chiralData =  readTextFile(chiralFile)
    return [ {'state': chirality, 
                 'CA': entry[0], 
                  'N': entry[1],
                  'C': entry[2],
                 'CB': entry[3],
                  'H': entry[4]}  for entry, chirality in zip(chiralData[0::2], chiralData[1::2]) ]
   

#def flipChirality(infile, chiralFile, outcoordsFile, outPDB):    
#    
#    # load the coords and inf from the pdb.
#    pdbAtoms = readAllAtoms(infile)
#    
#    chiralList = loadChiralCentres(chiralFile)
# 
#    for stereoCenter in chiralList:
#       if stereoCenter['state']=='T':
#           curResidue = getResidue(stereoCenter['CA'])
#           newResidue = flipResidueChirality( curResidue )
#            pdbAtoms = replaceResidueCoords( pdbAtoms, newResidue )
#
#    writeCoordsFile(pdbAtoms, outcoordsFile)
#    replacePdbAtoms(infile, newCoords, outPDB)

#def getResidue():
#    pass

    
def writeCoordsFile(atoms, filename):
    l = ['default_name']
    l.append(str(len(atoms)))
    for atom1, atom2 in zip( atoms[0::2], atoms[1::2] ):
        l_str = str(atom1[7]) + ' '
        l_str += str(atom1[8]) + ' '
        l_str += str(atom1[9]) + ' '
        l_str += str(atom2[7]) + ' '
        l_str += str(atom2[8]) + ' '
        l_str += str(atom2[9])
        l.append(l_str)
        
    writeTextFile(l, filename)
    

def eliminateCIS(infile):    
    # load the molecular info from the canon pdb.
    canonicalPdbAtoms = readAllAtoms(infile)

    # find the pdb with the lowest number of cis 
    lowestNumCis, lowestPdb = findBestCoords("coords.*.pdb")
 
    # keep the lowest pdb
    os.system("cp " + lowestPdb + " lowest.pdb")
    os.system("cp " + lowestPdb[0:-4] + ".rst lowest.rst")
    os.system("cp " + lowestPdb + ".ct lowest.ct")

    # loop while there isn't a minimum with zero cis states
    while lowestNumCis>0:

        # always start each run with the lowest inpcrds yet
        os.system("cp lowest.rst coords.inpcrd")

        # get the list of atoms that are in the cis state in the lowest coords file
        lowestCISAtomgroupsList = checkFileForCIS("lowest.ct")

        # make an atomGroups file for those cis bonds only. use the canonical pdb to help out but any pdb would do.
        makeAtomGroupsFile(lowestCISAtomgroupsList, canonicalPdbAtoms)

        print("Calling CUDAGMIN")

        # call the CUDAGMIN operation. The current atoms groups will try to spin a bunch of times.
        os.system( "CUDAGMIN" )

        # find the pdb with the lowest number of cis 
        newLowestNumCis, newLowestPdb = findBestCoords("coords.*.pdb")
 
        # check to see if the pdb with the lowest number of cis is lower than the current lowest
        if newLowestNumCis < lowestNumCis:
            # if so then keep it.
            lowestNumCis = newLowestNumCis
               
            os.system("cp " + newLowestPdb + " lowest.pdb")
            os.system("cp " + newLowestPdb[0:-4] + ".rst lowest.rst")
            os.system("cp " + lowestPdb + ".ct lowest.ct")
    
        print("lowest on this run: " + str(newLowestNumCis) + " lowest so far: " + str(lowestNumCis))


def createCTFile(infile, params):
    
    pdbAtoms = readAllAtoms(infile)
    
    # load the data
    initCTFile = readTextFile(params[0])


    # parse the atom and CTStates from the data
    peptideBondAtomsList = [ line for line in initCTFile[0::2] ] 

    # generate output list
    outList = []
 

    # initialise counter to count the c states
    c_Count = 0
   
    # loop through the peptide bonds;  O C N H in the order of.
    for peptideBondAtoms in peptideBondAtomsList:
        atomIndices = [ int(val) for val in peptideBondAtoms.strip('\r\n').split() ]
        OAtom = pdbAtoms[atomIndices[0] - 1]
        CAtom = pdbAtoms[atomIndices[1] - 1]
        NAtom = pdbAtoms[atomIndices[2] - 1]
        HAtom = pdbAtoms[atomIndices[3] - 1]
        OVec = np.array([ OAtom[7], OAtom[8], OAtom[9] ] )
        CVec = np.array([ CAtom[7], CAtom[8], CAtom[9] ] )
        NVec = np.array([ NAtom[7], NAtom[8], NAtom[9] ] )
        HVec = np.array([ HAtom[7], HAtom[8], HAtom[9] ] )        

        dihedral = computeDihedral(OVec, CVec, NVec, HVec)[0]

        bondState = ' N'
        if np.abs(dihedral)<float(params[1]):
            bondState = ' C'
            c_Count += 1
        if np.abs(dihedral)> (180.0 - float(params[1])):
            bondState = ' T'

        outList.append(peptideBondAtoms)
        #outList.append(bondState + " " + str(dihedral) + '\n')
        outList.append(bondState + '\n')

    writeTextFile(outList, infile + '.ct')
                
    return c_Count



# pinched these next few functions from Utilities. COuoldn't get all the different versions of python to talk to Utilities. Whatever.  

def generateTNBVecXYZ(genTNB, beta, alpha):
    # This function returns a unit vector pointing in the direction 
    # in the TNB frame defined by alpha and beta.
    # The vector is returned in the XYZ coords of the frame in 
    # which the TNB vectors are defined. 
     
    # Alpha is the azimuthal angle about the T vector (first in TNB list). 
    # Alpha = 0 is in the direction of the B vector (third in list)
    # Beta is the elevation angle which is zero in the B-N plane.
    # When the TNB frame is defined by the last three monomers in a polymer chain, 
    # Then alpha is the dihedral angle defined by the three monomers and the newly picked vector.
    # Beta is the angle between the new "bond" and the penultimate "bond".
    posTNB = bondAngleDihedral2TNB(np.array([1.0, beta, alpha]))
    return TNB2XYZ(genTNB, posTNB)

def bondAngleDihedral2TNB(pos): # given r, beta, alpha in terms of bond angle (zero to pi) and dihedral(-pi to pi)
    # return a cartesian vector in the TNB convention.  X->B, y-> N, and Z->T.  See book 6 page 79. 
    cartVect = sphericalPolar2XYZ([pos[0], pos[1] - np.pi/2, pos[2]])
    return np.array([cartVect[2], cartVect[1], cartVect[0]]) 

def TNB2XYZ(TNBframe, posTNB):
    # given a TNB frame defined in XYZ coords, and a TNB vector, posTNB, defined in the TNB frame, 
    # return the XYZ coords of posTNB 
    return np.inner(np.transpose(TNBframe), posTNB)

def sphericalPolar2XYZ(pos):
    # takes [r, theta, phi] numpy array. Computes the x,y,z coords on unit sphere and 
    # scales it to radius r thus returning x, y, z position of spherical polar input
    unitSpherePos = polarToUnitSphereXYZ(pos[1], pos[2]) 
    return pos[0] * unitSpherePos   

def polarToUnitSphereXYZ(theta, phi):
    # take theta and phi and computes the x, y and z position of the point on unit sphere.
    # this (and inverse XYZ2SphericalPolar) is the only place where this transformation is defined.
    # theta is from -pi/2 (south pole) to pi/2 (north pole), measured from xy plane.
    # phi is from -pi to pi. zero at +ve x axis.  
    return np.array([np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), np.sin(theta)]) 

def constructTNBFrame(p1, p2, p3):
    # given three arbitrary points defined in the lab XYZ frame, this function constructs 
    # three orthogonal vectors (T, N and B). T points from the second seed point in the list to the third seed point.
    # The N axis is normal to the plane formed by the three points, and is therefore orthogonal to T.
    # B is the cross product of T and N. The three unit vectors in these directions are returned as a numpy array of numpy arrays (a matrix)
    # returns None if p1, p2 and p3 are colinear
    
    # assume failure
    outVecs=None
    
    T = p3 - p2  # equivalent of b2 in bond angle and dihedral functions
    tHat = T/np.linalg.norm(T)

    U = p2 - p1 # equivalent of b1 in bond angle and dihedral functions
    uHat = U/np.linalg.norm(U)
    
    n = np.cross(uHat, tHat)  # equiv of n1 = b1 x b2 in dihedrals
    nMag =  np.linalg.norm(n)
    
    if ( abs(nMag - 0.0) > 1e-10):  # check for colinearity (cross product of T and U is zero)
        # now we're not colinear can divide by norm(n)
        nHat= n/nMag
        
        b = np.cross(nHat, tHat)  # equiv of m1 = n1Hat x b2Hat in dihedral reconstruction 
        bHat = b/np.linalg.norm(b) # just in case
        outVecs = np.array([tHat, nHat, bHat])
    
    return outVecs


def groupAtomsXYZ(infile, outfile):
    # read in the atoms from the pdb
    atoms=readAllAtoms(infile)
    
    # re label atoms as C, H, N, O, P, S to reduce number of objects in blender

    # convert the atom type to an atom colour that reflects the functional gorup of the residue.
    for atom in atoms:
        if atom[3]=='LIG':
            atom[1] ='W'
        elif 'N' in atom[1]:
            atom[1] ='N'
        elif 'C' in atom[1]:
            atom[1] ='C' 
        elif 'O' in atom[1]:
            atom[1] ='O' 
        elif 'H' in atom[1]:
            atom[1] ='H' 
        elif 'S' in atom[1]:
            atom[1] ='S' 
        elif 'P' in atom[1]:
            atom[1] ='P' 
        else:
            print('Unrecognised atom', atom[1], ' line:', atom[0])
            atom[1] ='Pb' # dark grey

    writeXYZ(atoms, outfile)
    print('Done')
    return          



def makeXYZForBlender(infile, backboneOnly, resMode, outfile):
    # read in the atoms from the pdb
    atoms=readAllAtoms(infile)
    if backboneOnly==1:
        # filter out the backbone atoms
        atom2XYZ = [ atom for atom in atoms if atom[1] in ['CA', 'C', 'N', 'NX']]
        for atom in atom2XYZ:
            if atom[1]=='CA':
                atom[1]='C'
    elif backboneOnly==2:
        atom2XYZ = [ atom for atom in atoms if atom[1] in ['CA', 'NX'] ] 
        for atom in atom2XYZ:
            if atom[1]=='CA':
                atom[1]='C'
    else:
        atom2XYZ = atoms
            

    if resMode==2:
        # convert the atom type to an atom colour that reflects the functional gorup of the residue.
        for atom in atom2XYZ:
            if atom[3] in ['ARG', 'HIS', 'LYS']:
                atom[1] ='P' # make the positive charge residues gold
            elif atom[3] in ['ASP', 'GLU']:
                atom[1] ='O' # make the negative charge residues brown
            elif atom[3] in ['SER', 'THR', 'ASN', 'GLN']:
                atom[1] ='C' # make the polar residues purple
            elif atom[3] in ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']:
                atom[1] ='N' # make the hydrophobic residues blue
            elif atom[3] in ['CYS', 'SEC','GLY', 'PRO', 'HYP']:
                atom[1] ='S' # make the special residues green
            elif atom[3] in ['Imb', 'Ime', 'Imc']:
                atom[1] ='Ca' # make the PLP backbones some color
            else:
                print('Unrecognised residue name:', atom[3], ' line:', atom[0])
                atom[1] ='Pb' # dark grey

    if resMode==1:
        # convert the atom type to an atom colour that reflects the amino acid
        for atom in atom2XYZ:
            if atom[3]=='ARG':
                atom[1] = 'H'
            elif atom[3]=='HIS':
                atom[1] = 'He'
            elif atom[3]== 'LYS':
                atom[1] = 'Li'
            elif atom[3]== 'ASP':
                atom[1] = 'Be'
            elif atom[3]== 'GLU':
                atom[1] = 'B'            
            elif atom[3]== 'SER':
                atom[1] = 'C'
            elif atom[3]== 'THR':
                atom[1] = 'N'            
            elif atom[3]== 'ASN':
                atom[1] = 'O'
            elif atom[3]== 'GLN':
                atom[1] = 'F'
            elif atom[3]== 'ALA':
                atom[1] = 'Ne'
            elif atom[3]== 'VAL':
                atom[1] = 'Na'  
            elif atom[3]== 'ILE':
                atom[1] = 'Mg'  
            elif atom[3]== 'LEU':
                atom[1] = 'Al'
            elif atom[3]== 'MET':
                atom[1] = 'Si' 
            elif atom[3]== 'PHE':
                atom[1] = 'P'
            elif atom[3]== 'TYR':
                atom[1] = 'S'                
            elif atom[3]== 'TRP':
                atom[1] = 'Cl'
            elif atom[3]== 'CYS':
                atom[1] = 'Ar'
            elif atom[3]== 'SEC':
                atom[1] = 'K'
            elif atom[3]== 'GLY':
                atom[1] = 'Ca'
            elif atom[3]== 'PRO':
                atom[1] = 'Ga'                
            elif atom[3]== 'HYP':
                atom[1] = 'Ge'                
            elif atom[3] in ['Imb', 'Ime', 'Imc']:
                atom[1] ='As'
            else:
                print('Unrecognised residue name:', atom[3], ' line:', atom[0])
                atom[1] ='Pb' # dark grey


    writeXYZ(atom2XYZ, outfile)
    print('Done')
    return          

def writeXYZ(atoms, outfile):
    lines = [str(len(atoms))+'\n']
    lines.append('XYZ data written by pdbProc\n')
    for atom in atoms:
        lines.append( str(atom[1]) + ' ' + str(atom[7]) + ' ' + str(atom[8]) + ' ' + str(atom[9]) +'\n')

    writeTextFile(lines, outfile)


def parseXYZFile(xyz):
    ''' reads an xyz file and returns an array of frames. Each frame containes the list of atoms names and a list of corresponding coords as numpy coords.'''
    # read in the xyz file
    xyzData = readTextFile(xyz)

    # initialise frames output
    frames=[]

    curline=0
    while curline < len(xyzData):
        # read the frame length from the current line in xyzData (starts off at 0)
        framelen = int(xyzData[curline])

        #get the data for the current frame
        xyzList = xyzData[curline + 2 : curline + 2 + framelen]

        #increment the cur line index
        curline += 2 + framelen

        #create the output lists for the atom names and atom coords
        atomNames = []
        atomCoords = []
        for atomData in xyzList:
            atom = atomData.split()
            atomNames.append(atom[0])
            atomCoords.append(np.array([float(atom[1]),float(atom[2]),float(atom[3])]))
    
    
        frames.append([atomNames, atomCoords])

    return frames

def replacePdbXYZ(infile, xyz, outputfile):
    ''' Function reads in an xyz file and replaces the coords in the pdb with the coords in the xyz file.
        Generates a PDB for each structure in the xyz file.'''
    frameNum=0
    for frame in parseXYZFile(xyz):
        replacePdbAtoms(infile, frame[1], outputfile + '.' + str(frameNum) + '.pdb')
        frameNum += 1
    return


def replacePdbAtoms(infile, newCoords, outfile, pdb=None, atoms=None):
    ''' Creates a verbatim copy of the pdb infile but with the coords replaced by the list of coords in new Atoms.
        Assumes the coords are in the same order as the pdb file atom lines and are in a list of xyz triplets.
        works for a 2D numpy array of m x 3. '''

    if atoms==None:
        # read in the atoms from the pdb
        atoms=readAllAtoms(infile)

    if pdb==None:
        # get the raw pdb file
        pdb=readTextFile(infile)

    # check the lists are compatible
    if len(atoms)!=len(newCoords):
        print("atom lists incompatible sizes") 
        sys.exit(0)


    # seed the output array
    outputlines=[]
    curCoord=0

    # loop through the raw input
    for line in pdb:
        # copy the line for output
        newLine = cp.copy(line)

        # check to see if the current line in the pdb is an atom or a hetatom
        if line[0:4]=="ATOM" or line[0:6]=="HETATM":
            # if it is then parse the input atom line
            try:
                atom = parsePdbLine(newLine)
            except:
                print( "line: " + line + " not understood")
                exit(0)
            # replace the coords
            atom[7] = newCoords[curCoord][0]
            atom[8] = newCoords[curCoord][1]
            atom[9] = newCoords[curCoord][2]
 
            curCoord += 1
        
            # generate the new line
            newLine = pdbLineFromAtom(atom)
       
        outputlines.append(cp.copy(newLine))

    writeTextFile(outputlines, outfile)

    return


def getCoordsFromAtoms(atoms):
    coords = []
    for atom in atoms: 
        coords.append(atom[7])
        coords.append(atom[8])
        coords.append(atom[9])
    return coords


def replaceAtoms(atoms, newCoords):
    ''' Replaces the atomic coords in the atoms array with a new set of coords.'''

    if len(atoms)!=len(newCoords):
        print("atom lists incompatible sizes")
        sys.exit(0)

    newAtoms=[]
    # loop through the raw input
    for atom,newCoord in zip(atoms,newCoords):
        # copy the line for output
        newAtom=cp.copy(atom)

        # replace the coords
        newAtom[7] = newCoord[0]
        newAtom[8] = newCoord[1]
        newAtom[9] = newCoord[2]
         
        # append the data to the output array
        newAtoms.append(newAtom)

    return newAtoms


def parseAtomGroupsFile(filename):

    # read in the atom data
    atomGroupData=readTextFile(filename)

    # initialise the output array
    atomGroups = []
    curGroup = []
    rotAtomsAngle = []
    processingFirstGroup = 1
    
    # loop through the input data
    for line in atomGroupData:
        # parse the current line
        data=line.split()
        # if the current line not empty then process it or do nothing
        if len(data)>0:
            # if the line is a group statement we are creating a new group.
            # capture the rotation atom numbers
            if data[0]=='GROUP':
                # check to see if we are processing the first group.
                if processingFirstGroup==1:
                    # if so then capture the data about the rotation axis and angle
                    rotAtomsAngle=[int(data[2]), int(data[3]), float(data[5])]
                else:
                    # if not then store the information about the previous group
                    atomGroups.append([rotAtomsAngle, curGroup])
                    # set the flag indication we are not on the first group any more
                    processingFirstGroup=0
                    # capture the data about the rotation axis and angle of the new group.
                    rotAtomsAngle=[int(data[2]), int(data[3]), float(data[5])]
                    curGroup = []
            else:
                # not a group statement so append the atom number to the current group
                curGroup.append(int(data[0]))
 
    # append the last group to the output list   
    atomGroups.append([rotAtomsAngle, curGroup])

    return atomGroups


def rotateVector(point,axisPoint1,axisPoint2,angle):
    ''' function takes an axis defined by two points and rotates a third point about the axis by an angle given in degrees'''

    # translate all points relative to the axisPoint1 and normalise the unit vector along the axis.
    vector = point - axisPoint1
    axis = axisPoint2 - axisPoint1
    axisNorm = axis/np.linalg.norm(axis)

    # project vector onto axis
    axialComponent = np.dot(vector, axisNorm)
    # compute the radial component
    radialVector = vector - axialComponent * axisNorm
    # comput the length of the radialVector
    radialVectorMag = np.linalg.norm(radialVector)

    # compute the unit vectors in the radial direction and the binormal
    radialVectorNorm = radialVector/radialVectorMag
    binorm = np.cross(radialVectorNorm, axisNorm)

    # compute the new coordinates of the vector
    newComp1 = np.cos(angle * np.pi/180.0) * radialVectorMag
    newComp2 = np.sin(angle * np.pi/180.0) * radialVectorMag
    newVector = axialComponent * axisNorm + newComp1 * radialVectorNorm + newComp2 * binorm + axisPoint1

    #compute and return the final rotated vector and move it back to the original frame of reference
    return newVector

def rotateAtoms(atoms,group):
    ''' Takes the group of atoms defined in the group command and rotates each one about the axis defined in the group command by the angle defined in the group command'''
    # unpack the parameters from the group command - our array is zero based, whereas number is ith entry in list, so subtract 1
    axisPoint1 = group[0][0]-1
    axisPoint2 = group[0][1]-1
    angle = group[0][2]
    atomGroup = [ entry-1 for entry in group[1]]

    #define the rotation axis:
    point1 = np.array([atoms[axisPoint1][7], atoms[axisPoint1][8], atoms[axisPoint1][9]])
    point2 = np.array([atoms[axisPoint2][7], atoms[axisPoint2][8], atoms[axisPoint2][9]])
 
    # loop through the atom group and rotate each atom by angle about the defined axis.
    for atom in atomGroup:
        newAtomPos=rotateVector(np.array([atoms[atom][7], atoms[atom][8], atoms[atom][9]]),point1,point2,angle)
        atoms[atom][7] = newAtomPos[0]
        atoms[atom][8] = newAtomPos[1]
        atoms[atom][9] = newAtomPos[2]

    return atoms


def rotateGroup(infile,atomGroupsFilename,outfile):
    
    '''Function reads in the atoms groups data and rotates each group about the axis defined in that group by a set angle.
       The new coords are then written down.  The groups are processed in the order they appear in the atomgroups file.
       The numbers in the atom groups file refer to the order the atoms appear in the pdb'''
    print(outfile)
    # read in the atoms from a the file
    atoms = readAllAtoms(infile)

    # parse the atoms group instructions for rotations
    atomGroups = parseAtomGroupsFile(atomGroupsFilename)

    # construct the new atom coords for each group defined in the atom groups file
    for group in atomGroups:
        atoms = rotateAtoms(atoms, group)

    # write the output file with the new atom coords in it
    replacePdbAtoms(infile, extractCoords(atoms), outfile)

    return    
 
def extractCoords(atoms):
    return [ np.array([atom[7], atom[8], atom[9]]) for atom in atoms ]

   

def torsionDiff(indirectory, outfile):
    # in the current directory find the lowestN.1.pdb files
    print(indirectory, outfile)

    # for each pdb file compute the torsion


    # output the energy at the top of the file
    
    # output the 



    return

def breakChainIntoResidues(chain):

    # create the output array
    residues = []

    # set the current Residue to the value of the first residue in the chain.
    curRes = chain[0][5]
    # create the dummy variable to store a list of atoms for each residue.
    atomsInRes = []
    
    # go through the chain appending each atom to the dummy list.
    # if we find the first atom in a new residue then increment the residue number and add the previous residue to the outgoing list.
    # reset the list for the new current residue
    for atom in chain:
        if atom[5]!=curRes:
            curRes+=1
            residues.append(atomsInRes)
            atomsInRes=[]
        atomsInRes.append(atom)
    
    # append the last residue
    residues.append(atomsInRes)
    return residues

def ComputeHBondDistance(res1, res2):
    '''function extracts the H from the N H of res1 and the O from C=O of res2 and computes the length'''
    H = [np.array([atom[7], atom[8], atom[9]]) for atom in res1 if atom[1]=='H'][0]
    O = [np.array([atom[7], atom[8], atom[9]]) for atom in res2 if atom[1]=='O'][0]

    return np.linalg.norm(H-O)


def computePuckerAndChi(res):
    # compute chi's
    chi = computeChi(res)
    
    # computer the puckers return the pucker state from both methods 
    pucker1 = checkPuckerState(res)
    
    # determine pucker based on chi1.
    puckerChi = 'Endo'
    if chi[0]<0:
        puckerChi = 'Exo'
    
    #debug information both puckers should be the same.
    if pucker1!=puckerChi:
        print(pucker1, puckerChi, chi[0])
        print(res)
    
    return [pucker1,chi]

def computeChi(res):

#extract the relevant information from the residues and present as position vectors of atoms.
    # C=[np.array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='C'][0]
    N=[np.array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='N'][0]
    CA=[np.array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='CA'][0]
    CB=[np.array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='CB'][0]
    CG=[np.array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='CG'][0]
    CD=[np.array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='CD'][0]
    
#Only need the following to generate an AUX for Xi which is a weird dihedral from parker. Not so bothered.
#     if res[0][3]=='PRO':
#         HG3=[array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='HG3'][0]
#         HG2=[array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='HG2'][0]
#         
#         #Need to decide which HG to compute dihedral for - the one that gives 4R chirality
#         CDCG = CG-CD
#         CBCG = CG-CB
#         H2CG = CG-HG2
#         
#         N=cross(CDCG,CBCG)
#         N=N/linalg.norm(N)
#         CZ=vdot(H2CG,N)
#         AUX=HG2
#         if CZ<0:
#             #chirality4='S' when using HG2 so pick the other one
#             AUX=HG3
#     if res[0][3]=='HYP':
#         AUX=[array([atom[7],atom[8],atom[9]]) for atom in res if atom[1]=='OD1'][0]

    # compute Chis
    chi1 = computeDihedral(N,CA,CB,CG)
    chi2 = computeDihedral(CA,CB,CG,CD)
    chi3 = computeDihedral(CB,CG,CD,N)
    chi4 = computeDihedral(CG,CD,N,CA)
    chi5 = computeDihedral(CD,N,CA,CB)
     
    if chi1[1]==1:
        print('N,CA,CB,CG are colinear in residue: '+str(res[0][5]))
    if chi2[1]==1:
        print('CA,CB,CG, CD are colinear in residue: '+str(res[0][5]))
    if chi3[1]==1:
        print('CB,CG,CD,N are colinear in residue: '+str(res[0][5]))
    if chi4[1]==1:
        print('CG,CD,N,CA are colinear in residue: '+str(res[0][5]))
    if chi5[1]==1:
        print('CD,N,CA,CB are colinear in residue: '+str(res[0][5]))

    return [chi1[0],chi2[0],chi3[0],chi4[0],chi5[0]]

def computeTorsion(res0,res1,res2):

    #extract the relevant information from the residues and present as position vectors of atoms.
    C0 = [np.array([atom[7],atom[8],atom[9]]) for atom in res0 if atom[1]=='C'][0]
    N1 = [np.array([atom[7],atom[8],atom[9]]) for atom in res1 if atom[1]=='N'][0]
    CA1 = [np.array([atom[7],atom[8],atom[9]]) for atom in res1 if atom[1]=='CA'][0]
    C1 = [np.array([atom[7],atom[8],atom[9]]) for atom in res1 if atom[1]=='C'][0]
    N2 = [np.array([atom[7],atom[8],atom[9]]) for atom in res2 if atom[1]=='N'][0]
    CA2 = [np.array([atom[7],atom[8],atom[9]]) for atom in res2 if atom[1]=='CA']
    CA2Exists = 0
    if CA2:
        CA2Exists = 1    
        CA2 = CA2[0]
    
    #compute phi; the C0'-N1-Ca1-C1' dihedral. Requires res0 and res1
    phi = computeDihedral(C0,N1,CA1,C1)

    #compute psi; the N1-Ca1-C1'-N2 dihedral. Requires res1 and res2
    psi = computeDihedral(N1,CA1,C1,N2)

    #compute omega; the Ca1-C1-N2-CA2 dihedral. Requires res1 and res2
    if CA2Exists:
        omega = computeDihedral(CA1,C1,N2,CA2)
    else:
        omega = [0.0, 0]
        
    if phi[1]==1:
        print('C0,N1 and CA1 are colinear in residue: '+str(res1[0][5]))
    if psi[1]==1:
        print('CA1, C1 and N2 are colinear in residue: '+str(res1[0][5]))
    if omega[1]==1:
        print('C1, N2 and CA2 are colinear in residue: '+str(res1[0][5]))
    return [phi[0],psi[0],omega[0]]

#Yings definition of torsion angles
def calcTorsionangle(atom1, atom2, atom3, atom4):
   
    vec1 = atom2 - atom1
    vec2 = atom3 - atom2
    vec3 = atom4 - atom3
    return np.arctan2( np.vdot( np.linalg.norm(vec2)*vec1 , np.cross(vec2,vec3) ), np.vdot( np.cross(vec1,vec2), np.cross(vec2, vec3)) )*180/np.pi
    

# given four vectors compute the dihedral about the line connecting P2 to P3.
# checks for colinearity and returns the angle in degrees and a flag indicating 
# whether or colinearity was detected. IN the latter case the angle defaults to 0.
def computeDihedral(P1,P2,P3,P4):
 
    # construct the in plane vectors V1 and U1 are both plane V and U respectively. UV is in both.
    U1 = P2-P1
    U1 = U1/np.linalg.norm(U1)
    UV = P3-P2
    UV = UV/np.linalg.norm(UV)
    V1 = P4-P3
    V1 = V1/np.linalg.norm(V1)

    # compute the dot product between the input vectors
    COSU = np.vdot(U1,UV)
    COSV = np.vdot(V1,UV)

    # set the default output value
    theta = 0.0
    colinear = 0
    # check for colinearity (COSU=1 or COSV=1)
    if (1.0-abs(COSU))>1E-6 and (1.0-abs(COSV))>1E-6:
        # Compute the normals to the planes
        N1 = np.cross(U1, UV)
        N1 = N1/np.linalg.norm(N1)
        N2 = np.cross(UV, V1)
        N2 = N2/np.linalg.norm(N2)
        
        # compute the binormal to N1 and UV.
        M1 = np.cross(UV,N1)
        
        # compute the components of N2 in the frame N1,UV,M1. The component of N2 along UV is always zero by definition.
        COSTHETA = np.vdot(N1,N2)
        SINTHETA = np.vdot(M1,N2)

        # shave off rounding and precision errors. COSTHETA should never be higher than 1 since N1 and N2 are normalised -doesn't matter too much in atan2 function.
        if COSTHETA>1.0:
            COSTHETA=1.0

        # Use the atan2 function which gives correct sign and a range between -180 and 180 in degrees,
        theta = np.degrees(np.arctan2(SINTHETA,COSTHETA))
    else:
        colinear = 1

    return theta, colinear


def puckerBreakDown(infile):
    '''This function takes an infile and analyses it's pucker state and colPol state, and outputs this information per residue.'''
    #read all the atoms.
    atoms=readAllAtoms(infile)
    
    #identify the separate chains in the pdb atom data
    chains=breakAtomsInToChains(atoms)

    #loop through each chain
    colPos=[]
    puckerState=[]
    for chain in chains:
        puckerStateChain=checkPuckerList(chain)
        curChainColPos=[ idGXYPatternRes(puckerStateChain,residue[0]) for residue in puckerStateChain]
        colPos=colPos+curChainColPos
        puckerState=puckerState+puckerStateChain
    

    numXProEndo=0
    numXProExo=0
    numXHypEndo=0
    numXHypExo=0
    numYProEndo=0
    numYProExo=0
    numYHypEndo=0
    numYHypExo=0

    for residue,colPosRes in zip(puckerState,colPos):

        if residue[1]=='PRO':
            if colPosRes=='X':
                if residue[2]=='Endo':
                    numXProEndo+=1
                if residue[2]=='Exo':
                    numXProExo+=1                    
            if colPosRes=='Y':
                if residue[2]=='Endo':
                    numYProEndo+=1
                if residue[2]=='Exo':
                    numYProExo+=1                    

        if residue[1]=='HYP':
            if colPosRes=='X':
                if residue[2]=='Endo':
                    numXHypEndo+=1
                if residue[2]=='Exo':
                    numXHypExo+=1                    
            if colPosRes=='Y':
                if residue[2]=='Endo':
                    numYHypEndo+=1
                if residue[2]=='Exo':
                    numYHypExo+=1
    
    
    # TotalEndo = numXProEndo+numXHypEndo+numYProEndo+numYHypEndo
    # TotalExo = numXProExo+numXHypExo+numYProExo+numYHypExo
    # TotalProline = numXProEndo+numYProEndo+numXProExo+numYProExo
    # TotalHyp = numXHypEndo+numYHypEndo+numXHypExo+numYHypExo
    TotalX = numXProEndo+numXProExo+numXHypEndo+numXHypExo
    TotalY = numYProEndo+numYProExo+numYHypEndo+numYHypExo
    
    # TotalXEndo = numXProEndo+numXHypEndo
    # TotalYEndo = numYProEndo+numYHypEndo
    # TotalXExo = numXProExo+numXHypExo
    # TotalYExo = numYProExo+numYHypExo
    
    # TotalXPro = numXProExo+numXProEndo
    # TotalYPro = numYProExo+numYProEndo
    # TotalXHyp = numXHypExo+numXHypEndo
    # TotalYHyp = numYHypExo+numYHypEndo
    
    # TotalProEndo = numXProEndo+numYProEndo
    # TotalProExo = numXProExo+numYProEndo
    # TotalHypEndo = numXHypEndo+numXHypEndo
    # TotalHypExo = numXHypExo+numYHypExo
    
    Total = TotalX+TotalY
    
    print("numXProlineEndo: ",str(numXProEndo), "  Total %age: "+str(float(numXProEndo)/float(Total)) )
    print("numXProlineExo: ",str(numXProExo), "  Total %age: "+str(float(numXProExo)/float(Total)) )
    print("numYProlineEndo: ",str(numYProEndo), "  Total %age: "+str(float(numYProEndo)/float(Total)) )
    print("numYProlineExo: ",str(numYProExo), "  Total %age: "+str(float(numYProExo)/float(Total)) )
    print("numXHypEndo: ",str(numXHypEndo), "  Total %age: "+str(float(numXHypEndo)/float(Total)) )
    print("numXHypExo: ",str(numXHypExo), "  Total %age: "+str(float(numXHypExo)/float(Total)) )
    print("numYHypEndo: ",str(numYHypEndo), "  Total %age: "+str(float(numYHypEndo)/float(Total)) )
    print("numYHypExo: ",str(numYHypExo), "  Total %age: "+str(float(numYHypExo)/float(Total)) )
    
    
    
    return
  
def residueInfo(infile, outfile):
    atoms = readAllAtoms(infile)
  
    outputArray = generateResidueInformation(atoms)

    outputLines=[]
    
    #Res Id, chi1,chi2,chi3,chi4,chi5,phi,psi,omega,pucker
    for res in outputArray:
        l=str(res[0])+', '+str(res[6])+', '+str(res[7])+', '+str(res[8])+', '+str(res[1])+', '+str(res[2])+', '+str(res[3])+', '+str(res[4])+', '+str(res[5])+', '+res[9]+'\n'
        print(l )
        outputLines.append(l)
        
    #write the residue information to file
    writeTextFile(outputLines,outfile)
    
    return


def ramachandrans(infile, configFile):
    params = loadJson(configFile)
    
    # create a handle to a matplot lib figure of the right size
    fig = plt.figure(figsize=(params['figsizex'], params['figsizey']))
    
    rama3D = 0
    try:
        if params['rama3D']==1:
            ax = fig.add_subplot(projection='3d')
            rama3D = 1
    except KeyError:
        pass

    # loop through each file and extract the information to plot in the current figure
    legendLabels = []
    legendHandles = []
    
    try: 
        plotKeys = params['order']
    except KeyError:
        plotKeys = [ key for key in params['plotDetails'] ]
    
    for plotDictKey in plotKeys:
        # get the file name, sets to plot and marker information
        filename = params['plotDetails'][plotDictKey]['filename']
        subsets = params['plotDetails'][plotDictKey]['subsets']
        startRes = params['plotDetails'][plotDictKey]['startRes'] # start & end res mainly for filtering ACEs and NMEs
        endRes = params['plotDetails'][plotDictKey]['endRes']
        print("processing file: ", filename)
        
        # load file and generate Phi Psi angles for entire file
        atoms = readAllAtoms(filename)
        resIds, resNames, phi, psi= generatePhiPsi(atoms, filename[:-4] + '.rama')

        # cope with -ve index value for endRes index
        if endRes<0:
            endRes = len(phi) + endRes

        # loop through each subset and extract the requested information
        # subset is a dictionary whose keys are a res name to plot, and whose entries are the plotmarker
        for resName in subsets:
                style = subsets[resName]

                # figure out indices that match the specified conditions
                if resName=='All':
                    plotIndices = [ i for i in range(len(phi)) if ( resIds[i]>=startRes and resIds[i]<=endRes ) ]
                else:
                    plotIndices = [ i for i in range(len(phi)) if ( resNames[i]==resName and resIds[i]>=startRes and resIds[i]<=endRes ) ]
                
                
                if rama3D:
                    # plot the phi, psi points for the current subPlot
                    ax.scatter(np.array(phi)[plotIndices],
                               np.array(psi)[plotIndices],
                               np.array(resIds)[plotIndices],
                               marker=style['marker'],
                               s=style['markersize'],
                               c=style['markerfacecolor'])
                else:                
                    # plot the phi, psi points for the current subPlot                
                    pHandle, = plt.plot(np.array(phi)[plotIndices], 
                                        np.array(psi)[plotIndices], 
                                        markeredgewidth=int(style['markeredgewidth']),
                                        markerfacecolor=style['markerfacecolor'], 
                                        markeredgecolor=style['markeredgecolor'],
                                        marker=style['marker'], 
                                        markersize=style['markersize'],
                                        linestyle=style['linestyle'])
                
                
                    if len(params['plotDetails'])>1:
                        legendLabels.append(plotDictKey + " " + resName)
                    else:
                        legendLabels.append(resName)
    
                    legendHandles.append(pHandle)
 
    if not rama3D:
        if params['includeLegend']==1:
            plt.legend(legendHandles, legendLabels, loc=params['legendLoc'], fontsize=params['legendFontSize'], frameon=False)
            
        # set global params and behaviour for the figure
        plt.xlabel(params['xlabel'], fontsize=params['labelfontsize'])
        plt.ylabel(params['ylabel'], fontsize=params['labelfontsize'])
        title = params['title']
        plt.title(title, fontsize=params['titlefontsize'])
        plt.xticks(params['xticks'], fontsize=params['tickfontsize'])
        plt.yticks(params['yticks'], fontsize=params['tickfontsize'])
        plt.xlim([params['phiMin'], params['phiMax']])
        plt.ylim([params['psiMin'], params['psiMax']])

    plt.savefig(params['pngName'])
    plt.show()

    
def ramachandran(infile, configFile):
    
    params = loadJson(configFile)

    atoms = readAllAtoms(infile)
    
    _, _, phi, psi= generatePhiPsi(atoms, infile[:-4] + '.rama')
    
    createRamaPlot(phi, psi, infile[:-4]+'.png', params)


def createRamaPlot(phi, psi, filename, settings):
    plt.figure(figsize=(settings['figsizex'], settings['figsizey']))
    try:
        plotRanges = settings['plotRanges']
        labels = []
        for plotRange in plotRanges:
            labels.append(plotRange)
            s = plotRanges[plotRange]['start']
            e = plotRanges[plotRange]['end']
            pm = plotRanges[plotRange]['plotMarker']
            plt.plot(phi[s:e], psi[s:e], pm)
        plt.legend(labels)
    except KeyError:
        plt.plot(phi, psi, settings['plotMarker'])
    plt.xlabel(settings['xlabel'], fontsize=settings['labelfontsize'])
    plt.ylabel(settings['ylabel'], fontsize=settings['labelfontsize'])
    title = settings['title']
    try:
        if settings['includeFilenameInTitle']:
            title = settings['title'] + " " + filename 
    except KeyError:
        pass
    plt.title(title, fontsize=settings['titlefontsize'])
    plt.xlim([settings['phiMin'], settings['phiMax']])
    plt.ylim([settings['psiMin'], settings['psiMax']])
    plt.xticks(fontsize=settings['tickfontsize'])
    plt.yticks(fontsize=settings['tickfontsize'])
    try:
        outFilename = settings['pngName']
    except KeyError:
        outFilename = filename
    plt.savefig(outFilename)
    plt.show()
    return


def generatePhiPsi(atoms, outfile):

    resids, resnames, phis, psis = computePhiPsi(atoms)

    outputLines=[]
    
    #Res Id, phi, psi
    for res, resname, phi, psi in zip(resids, resnames, phis, psis):
        l = str(res) + ', ' + str(resname) + ', ' + str(phi) + ', ' + str(psi) + '\n'
        outputLines.append(l)
        
    #write the residue information to file
    writeTextFile(outputLines, outfile)


    return resids, resnames, phis, psis

def computePhiPsi(atoms):

    # identify the separate chains in the pdb atom data
    chains = breakAtomsInToChains(atoms)

    resIds = []
    resNames = []
    phis = []
    psis = []
    
    # loop through each chain
    atomsInResiduesInChainList=[]
    for chain in chains:
        # create a list of groups of atoms which are grouped into residues, which in turn are grouped into chains
        atomsInResiduesInChainList.append( breakChainIntoResidues(chain) )
        
    # Loop through the list of atoms in residues in chains lists. first consider each chain in turn.
    for residueList in atomsInResiduesInChainList:
        
        # loop through all the residues in the chain starting from res = 2 to res = N - 1.
        for resIndex in range(1, len(residueList) - 1  ):
            #extract the indices of three adjacent residues
            prevRes = resIndex-1
            curRes = resIndex
            nextRes = resIndex+1

            #compute the psi, phi and omega torsion angles for each residue
            torsionAngles = computeTorsion( residueList[prevRes], residueList[curRes], residueList[nextRes])
            
            #write out the output line
            resIds.append(residueList[curRes][0][5])
            resNames.append( residueList[curRes][0][3] )
            phis.append( torsionAngles[0] )
            psis.append( torsionAngles[1] )

          
    return resIds, resNames, phis, psis



def generateResidueInformation(atoms):

    #identify the separate chains in the pdb atom data
    chains=breakAtomsInToChains(atoms)

    #loop through each chain
    outputArray=[]
    atomsInResiduesInChainList=[]
    for chain in chains:
        #create a list of groups of atoms which are grouped into residues, which in turn are grouped into chains
        atomsInResiduesInChainList.append(breakChainIntoResidues(chain))
        
    #Loop through the list of atoms in residues in chains lists. first consider each chain in turn.
    curChain=0
    for residues in atomsInResiduesInChainList:
        if curChain==0:
            nextChain=2
        if curChain==1:
            nextChain=0
        if curChain==2:
            nextChain=1
        curChain+=1
        
        #loop through all the residues in the chain starting from res=2 to res=N-1.
        #Use the C and N or the end termini to compute all the torsion angles we're interested in.
        numRes=len(residues)
        for resIndex in range(1,numRes-1):
            #extract the indices of three adjacent residues
            prevRes=resIndex-1
            curRes=resIndex
            nextRes=resIndex+1

            #compute the index of the ProX - essentially the current Residue +/-1 but in the next chain round
            ProXIndex=prevRes-1
            if nextChain==2:
                ProXIndex=nextRes
            
            #compute the psi, phi and omega torsion angles for each residue
            torsionAngles=computeTorsion(residues[prevRes],residues[curRes],residues[nextRes])
        
            
            #set some default outputs
            HBondDist=0.0
            chi=[0,0,0,0,0]#set default for non-pro/hyp residues
            puckerState='None' #assume no pucker
            
            if residues[curRes][0][3] in ['HYP','PRO']:
                #compute dihedral angles for each proline or hydroxyproline residue and also find the pucker state
                [puckerState,chi]=computePuckerAndChi(residues[curRes])
                #chi = chi1[0],chi2[0],chi3[0],chi4[0],chi5[0]]

            #if we're a gly not in the last group then find the H-Bond length to the C=O of the next P residue in the next chain along
            if residues[curRes][0][3] in ['GLY'] and curRes<35:
                #compute distance between curResidue H in NH and the O of the associated Pro - X residue C-0 in the other chain.
                HBondDist = ComputeHBondDistance(residues[curRes],atomsInResiduesInChainList[nextChain][ProXIndex])
                
            #write out the output line
            outputArrayLine=[residues[curRes][0][5],chi[0],chi[1],chi[2],chi[3],chi[4], torsionAngles[0], torsionAngles[1], torsionAngles[2], puckerState, HBondDist]

            #Res Id, chi1,chi2,chi3,chi4,chi5,phi,psi,omega,pucker, HBondDist
            outputArray.append(outputArrayLine)
    return outputArray

def checkTorsionNoFiles(atoms):

    #identify the separate chains in the pdb atom data
    chains=breakAtomsInToChains(atoms)

    #loop through each chain
    outputLines=[]
    outputArray=[]
    for chain in chains:

        #find the residues in the current chain and group all the atoms in each residue in a sub list.
        residues=breakChainIntoResidues(chain)
        
        #loop through all the residues in the chain starting from res=2 to res=N-1.
        #Use the C and N or the end termini to compute all the torsion angles we're interested in.
        numRes=len(residues)
        for resIndex in range(1,numRes-1):
            #extract the indices of three adjacent residues
            prevRes=resIndex-1
            curRes=resIndex
            nextRes=resIndex+1

            #compute the psi and phi torsion angles for each residue
            torsionAngles=computeTorsion(residues[prevRes],residues[curRes],residues[nextRes])

            #for the current residue compute the puckerState, None, Endo or Exo
            puckerState=checkPuckerList(residues[curRes])

            #write out the output line
            outputArrayLine=[residues[curRes][0][5], residues[curRes][0][3],puckerState[0][2],torsionAngles[0],torsionAngles[1]]

            outputLine=str(residues[curRes][0][5])+'\t'+residues[curRes][0][3]+'\t'+str(puckerState[0][2])+'\t'+str(torsionAngles[0])+'\t'+str(torsionAngles[1])+'\n'

            outputLines.append(outputLine)
            outputArray.append(outputArrayLine)

    
    return [outputLines,outputArray]
    
def checkTorsion(infile,outfile):
    '''wrapper for checking torsion as part of the collatePucker routine involving file IO.'''
    #read in the data
    #rawData=readTextFile(infile)
    atoms=readAtoms(infile)

    #call the no files version of the function for the classifyMinima routine 
    [outputLines, outputArray]=checkTorsionNoFiles(atoms)
    
    #write the torsion and pucker information to file
    writeTextFile(outputLines,outfile)

    return outputArray

def idGXYPatternAll(resList):
    
    #get the id of the first residue
    firstRes=idGXYPattern(resList)
    
    outList=[firstRes]
    curRes=firstRes
    #zip through the rest of the list incrementing the residue appending a G, X or Y in turn
    for _ in resList[1:]:
        if curRes=='G':
            curRes='X'
        else:
            if curRes=='X':
                curRes='Y'
            else: 
                curRes='G'
        
        outList.append(curRes)
    
    return outList
    
    
#format of res list is [[1,'PRO'],[2,'GLY'],... etc ]
def idGXYPatternRes(resList,res):
    '''Computes the collagen GXY pattern of the specified residue in the given resList. Assumes resList is a single chain only'''
    #ID the GXY type of the first residue in the list
    firstRes=idGXYPattern(resList)
    
    #get the pattern of the first three in the chain sorted out
    if firstRes=='G':
        GXY=['G','X','Y']
    if firstRes=='X':
        GXY=['X','Y','G']
    if firstRes=='Y':
        GXY=['Y','G','X']
        
    #figure out where we are in the chain number wise - assume chain resIDs are continuous.
    curPos=0
    for curRes in resList:
        if curRes[0]==res:
            break
        curPos+=1
    
    #figure out the remainder of dividing the position by three and use that to index the right string
    GXYType=GXY[curPos % 3]
    
    return GXYType

# identify the res type of the first residue in the collagen chain in the GXY 
# pattern.
#input is a list of residues in which the second entry is the type.
#e.g. [ [1,'PRO'.,...],[2,'HYP',...],....]
#output is a 'G','X',or 'Y' depending on the glycine content of the chain.
#intended for collagen strands only really.
def idGXYPattern(chain):

    GXYFound = 0
    lastGlyIndex = -1
    glyCount = 0
    curIndex = 0
    # loop through the chain until we have identified the GXY pattern
    while GXYFound == 0:
        # get the residue type of the curIndex
        curResType = chain[curIndex][1]

        # check for glycine residue
        if curResType == 'GLY':
        
            # if the current gly residue is exactly 3 away from the last one increment gly count.
            if (curIndex - lastGlyIndex) == 3:
                glyCount += 1
    
            # if the current index is more than 3 from the last glyindex then reset the glycount.
            if (curIndex - lastGlyIndex) > 3:
                glyCount = 0
    
            # record the current indexnumber as the last known glycine
            lastGlyIndex = curIndex
    
            # if we have encountered a gly three away from the last gly on 3 
            # successive occassions we have probably found the pattern...
            if glyCount == 3:
                GXYFound = 1
    
        # increment the current index
        curIndex += 1

        # sanity check to ensure we don't loop forever 
        if curIndex == len(chain):
            sys.exit('Unable to find GXY pattern')

    # we know that the lastGlyIndex is a G in the GXY pattern, so figure out what first res
    # in chain would be (index mod 3)
    #   X Y G     G     G
    #   0 1 2 3 4 5 6 7 8

    resType = 'G'
    if lastGlyIndex % 3 == 1:
        resType = 'Y'
    if lastGlyIndex % 3 == 2:
        resType = 'X'

    return resType

#compare the pucker pattern against the standard collagen pucker pattern.
#returns true if its a perfect match and false if it isn't.
#input is a a puckerStateList from the function
#e.g. [[1,'PRO','Endo'],[2,'HYP','Exo'],...]=checkPuckerList(Atoms)
def checkPuckerPattern(puckerStateList):
    #assume that the pattern is true
    inPattern=True

    #Identify where in the pattern the first residue is
    curResType=idGXYPattern(puckerStateList)    

    #loop through each residue and see if it causes the pattern to fail
    for res in puckerStateList:

        #only interested in the HYP or PRO residues; ignore everything else
        if res[1] in ['HYP','PRO']:

            #if the curResType is an X then the ring should be Endo so return false if its exo.
            if curResType=='X' and res[2]=='Exo':
                inPattern=False
            #if the curResType is a Y then the ring should be Exo so return false if its endo.
            if curResType=='Y' and res[2]=='Endo':
                inPattern=False

        #increment the residue type
        if curResType=='G':
            curResType='X'
        else:
            if curResType=='X':
                curResType='Y'
            else:
                curResType='G'

    return inPattern

def checkPuckerPatternWrapper(infile):
    #rawData=readTextFile(infile)
    atoms=readAtoms(infile)
    chains=breakAtomsInToChains(atoms)

    patternFlagList=[]
    for chain in chains:
        puckerStateList=checkPuckerList(chain)
        patternFlagList.append(checkPuckerPattern(puckerStateList))

    print('File: '+infile+' collagen Pucker Pattern:' )
    print(patternFlagList )

    #if one of the chains does not obey the pattern the whole protein doesn't/
    patternFlag=True
    for a in patternFlagList:
        if a==False:
            patternFlag=False
    
    return patternFlag

def parsePdbLine(line):
    l=line
    atom_seri = int(l[6:11])
    atom_name = l[12:16].split()[0]
    alte_loca = l[16]
    resi_name = l[17:20].split()[0]
    chai_iden = l[21]
    resi_numb = int(l[22:26])
    code_inse = l[26]
    atom_xcoo = float(l[30:38])
    atom_ycoo = float(l[38:46])
    atom_zcoo = float(l[46:54])
    try:
        atom_occu = float(l[54:60])
    except:
        atom_occu=0.0

    try:
        atom_bfac = float(l[60:66])
    except:
        atom_bfac=0.0    
    
    try:
        seg_id = l[72:76]
    except:
        seg_id=' '

    try:
        atom_symb = l[76:78].split()[0]
    except:
        try:
            atom_symb = l[68]
        except:
            atom_symb= ' '

    try:
        charge=l[78:80]
    except:
        charge=' '

    return [atom_seri, atom_name, alte_loca, resi_name, chai_iden, resi_numb, code_inse, atom_xcoo, atom_ycoo, atom_zcoo, atom_occu, atom_bfac,seg_id,atom_symb,charge]


#looks for start of chain with the ACE group and end of chain with the NME group
def breakAtomsInToChains(atoms):
    '''breaks an atom into chains if the chain letter increments or encounters an ACE following an NME'''
    lastAtomRes='First' #initialise lastAtomRes
    lastAtomChain=atoms[0][4]
    chains=[[]]
    for atom in atoms:
        #if this is at the point where the last atom was an NME and the current Atom is an ACE
        #then create a new element in the chains list. Or if the chain changes name.
        if ((atom[3]=='ACE') and (lastAtomRes=='NME')) or (lastAtomChain!=atom[4]):
            chains.append([])

        #append the atom to the last element in the chains list.
        chains[-1].append(atom)

        #make a note of the last atom residue and last chain
        lastAtomRes=atom[3]
        lastAtomChain=atom[4]

    return chains

def readAtoms(filename):
    pdb=readTextFile(filename)
    atoms=extractAtomsFromPDB(pdb)
    return atoms

def readAllAtoms(filename):
    pdb=readTextFile(filename)
    atoms=extractAllAtomsFromPDB(pdb)
    return atoms


def pdbLineFromAtom(atom):
    try:
        l='ATOM {: >06d} {: <4}{:1}{:3} {:1}{: >4d}{:1}   {: >8.3f}{: >8.3f}{: >8.3f}{: >6.2f}{: >6.2f}      {: <4}{: >2}{: >2}\n'.format(int(atom[0]), atom[1], atom[2], atom[3], atom[4], int(atom[5]), atom[6], float(atom[7]), float(atom[8]), float(atom[9]), float(atom[10]), float(atom[11]), atom[12],atom[13],atom[14])
    except:
        print("unable to write atom to string: " )
        print(atom )
        exit(0)
    return l

def writeAtomsToTextFile(atoms, filename):
    #open data file
    try:
        vst = open(filename, 'w')
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror) )
        raise Exception( "Unable to open output file: " + filename )

    #parse data 
    for atom in atoms:
        l=pdbLineFromAtom(atom)
        vst.write(l)
    vst.close()
    return


def loadJson(filename):
    with open( filename ) as f:
        data = json.load(f)
    return data


def readTextFile(filename):
    #read line data in from file
    try:
        vst = open(filename, 'r')
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror) )
        raise Exception( "Unable to open input file: " + filename )
    lines = vst.readlines()
    vst.close()
    return lines

def writeTextFile(lines, filename):
    #write line data to file
    try:
        vst = open(filename, 'w')
    except:
        raise Exception( "Unable to open output file: " + filename )
    for line in lines:
        a=line
        vst.write(a)
    vst.close()
    return

def extractAllAtomsFromPDB(lines):
    #parse data 
    atoms = []
    for line in lines:
        if line[0:4]=="ATOM" or line[0:6]=="HETATM":
            try:
                a = parsePdbLine(line)
                if not a in atoms:
                    atoms.append(a)
            except:
                print("line: " + line + " not understood" )
                exit(0)
    return atoms


def extractAtomsFromPDB(lines):
    #parse data 
    atoms = []
    for line in lines:
        if line[0:4]=="ATOM":
            try:
                a = parsePdbLine(line)
                if not a in atoms:
                    atoms.append(a)
            except:
                print("line: " + line + " not understood" )
                exit(0)
    return atoms


def removeDuplicateAtoms(infile, outfile):

    # load the data
    lines = readTextFile(infile)

    # initialise variables
    newPDB = []
    prevLineAdded = ''
    curAtomNum = 1

    #loop through each line in the pdb file
    for line in lines:
        # assume we are adding the line
        addLine = 1

        # check to see if it is an Atom
        if line[0:4]=='ATOM':
          
            #extract the atomic information
            curAtom = parsePdbLine(line)

            # scan through the output pdb file to see if the atom has already been added
            for outputLine in newPDB:
                # if the line from the output pdb is an atom then parse the data
                if outputLine[0:4]=='ATOM':
                    outputAtom = parsePdbLine(outputLine)
 
                    # if the xyz positions are the same then the curAtom has already been output
                    # so do not output the curAtom.
                    if (outputAtom[7]==curAtom[7]) and (outputAtom[8]==curAtom[8]) and (outputAtom[9]==curAtom[9]):
                        addLine=0

            # check we're still adding the line, if so then modify the atomic number
            if addLine==1:
                curAtom[0] = curAtomNum
                curAtomNum += 1 # ready for next one
            
                # convert atom to text line
                line = pdbLineFromAtom(curAtom)

        elif line[0:3]=='TER':
            if prevLineAdded=='TER':
                addLine = 0

        #if we're still adding the line after all that then add it...
        if addLine==1:
            newPDB.append(line)
            prevLineAdded=line[0:3]

    # output final data to text
    writeTextFile(newPDB, outfile)

    return




def removeLines(lines, items):
    newPDB=[]
    lastLineAdded=''
    numLinesRemoved = 0
    for line in lines:
        addLine = 1
        if line[0:4]=='ATOM':
            atom = parsePdbLine(line)
            # print(atom )
            for item in items:
                a = item.split()
                # print(a, atom[int(a[1])]
                if a[0] in atom[int(a[1])]:
                    addLine=0
        elif line[0:3]=='TER':
            if lastLineAdded=='TER':
                addLine=0
        else:
            lastLineAdded = ''
        if addLine==1:
            newPDB.append(line)
            lastLineAdded = line[0:3]
        else:
            numLinesRemoved += 1
    print("Number of lines removed: ", numLinesRemoved )
    return newPDB


def findResidues(atoms):
    residueList=[]
    for atom in atoms:
        if not [atom[5],atom[3]] in residueList:
            residueList.append([atom[5],atom[3]])
    return residueList

def getAtomsInResidueAll(atoms,res):
    atomsOut=[]
    for atom in atoms:
        if atom[5] == res:
            atomsOut.append(atom)
    return atomsOut


def getAtomsInResidue(atoms,res):
    atomsOut=[]
    for atom in atoms:
        if atom[5] == res:
            atomsOut.append(atom[0])
    return atomsOut


def findAtom(atomType,atoms,res):
    atomOut=[]
    for atom in atoms:
        if atom[5] == res:
            if atomType==atom[1]:
                atomOut=atom[0]
    return atomOut


# generates a list of residues in each chain
def findChains(pdbInfo):
    chainList = []
    curChain = []
    for line in pdbInfo:
        if line[0:3]=='TER':
            chainList.append(curChain)
            curChain = []
        elif line[0:4]=='ATOM':
            atom = parsePdbLine(line)
            if not [atom[5],atom[3]] in curChain:
                curChain.append([atom[5],atom[3]])
    return chainList


#routine for checking the state of a residue - assumes this is a proline or hydroxyproline
#and that the residue is a list of the atoms which belong to a only single residue.
def checkPuckerState(residue):

    NPOS=[np.array([atom[7],atom[8],atom[9]]) for atom in residue if atom[1]=='N'][0]
    CAPOS=[np.array([atom[7],atom[8],atom[9]]) for atom in residue if atom[1]=='CA'][0]
    CBPOS=[np.array([atom[7],atom[8],atom[9]]) for atom in residue if atom[1]=='CB'][0]
    CGPOS=[np.array([atom[7],atom[8],atom[9]]) for atom in residue if atom[1]=='CG'][0]
    # CDPOS=[np.array([atom[7],atom[8],atom[9]]) for atom in residue if atom[1]=='CG'][0]
    CPOS=[np.array([atom[7],atom[8],atom[9]]) for atom in residue if atom[1]=='C'][0]

    #Method 1 - Chi1 - N - Calpha - CBeta forms a plane. IF CGamma and C are on same 
    #sides of plane then we are Endo - same as Chi1 is +ve/-ve.
    puckerState='Exo'
    NCA = CAPOS-NPOS
    CACB = CBPOS-CAPOS
    n = np.cross(NCA,CACB)
    n = n/np.linalg.norm(n)
    NCG = CGPOS-NPOS
    NC = CPOS-NPOS
    CGZ = np.vdot(NCG,n)
    CZ = np.vdot(NC,n)
    if CGZ*CZ > 0:
        puckerState='Endo'

    return puckerState

def checkPuckerList(atoms):
   
    #creates a list of the residue numbers and their type
    residues=findResidues(atoms)

    #print(residues )

    #initialise output list
    puckerList=[]

    #go through each residue
    for (residueNum,residueName) in residues:
 
        #print(str(residueNum)+' '+residueName )
 
        #generate a list of atoms in the current residue
        curResidues=[atom for atom in atoms if atom[5]==residueNum]

        #initialise pucker State
        puckerState=''

        #if residue is pro or a hyp call the checkPuckerState routine
        if residueName=='PRO' or residueName=='HYP':
            puckerState=checkPuckerState(curResidues)

        #add puckerState to List
        puckerList.append([residueNum,residueName,puckerState])

#    print(puckerList )
    return puckerList 

#convert the state of the residue to the specified pucker state
def convertPuckerState(residue,state):

    #check the state of the current pucker
    currentPuckerState=checkPuckerState(residue)

    #set the default output
    newResidue=[]
    DG = np.array([0,0,0])

    #check to see if the residue needs converting
    if currentPuckerState!=state[0]:
        #Find the N, CAlpha and CBeta to define the base plane.
        NPOS = [np.array([curAtom[7],curAtom[8],curAtom[9]]) for curAtom in residue if 'N'==curAtom[1]][0]
        CAPOS = [np.array([curAtom[7],curAtom[8],curAtom[9]]) for curAtom in residue if 'CA'==curAtom[1]][0]
        CBPOS = [np.array([curAtom[7],curAtom[8],curAtom[9]]) for curAtom in residue if 'CB'==curAtom[1]][0]
        CGPOS = [np.array([curAtom[7],curAtom[8],curAtom[9]]) for curAtom in residue if 'CG'==curAtom[1]][0]

        # compute vectors in the N-CA-CB plane
        NCA=CAPOS-NPOS
        CBCA=CBPOS-CAPOS

        # find n
        n = np.cross(NCA, CBCA)
        n = n/np.linalg.norm(n)

        # gamma relative to N (taken as origin)
        NCG = CGPOS-NPOS


        # find distance of gamma from plane:
        CGDIST = np.vdot(NCG,n)

         
        # reflect gamma into plane
        CGREF = NCG - 2.0 * CGDIST * n


        # Translate gamma back into lab frame
        CGNEW = CGREF + NPOS

        # find translation of gamma group
        DG = CGNEW - CGPOS
    

    # output all the atoms, translating those attached to the CG by DG, keeping their relative positions the same
    for atom in residue:
        newatom = cp.copy(atom)
        if atom[1] in ['CG','OD1','HG2','HG3']:
            atomPos = np.array([atom[7],atom[8],atom[9]])+DG
            newatom[7] = atomPos[0]
            newatom[8] = atomPos[1]
            newatom[9] = atomPos[2]

        newResidue.append(newatom)

    return newResidue

def convertPdbPucker(pdb,puckerState):
    #generate list of unique residues for which the pucker state is specified in the file
    puckerResiduesList=[]
    for puckerRes in puckerState:
        if not puckerRes[0] in puckerResiduesList:
            puckerResiduesList.append(puckerRes[0])

    #copy the old pdb across one line at a time into a new array.
    #if we are processing an atom line and the atom belongs to a residue in the puckerResidueslist 
    #then extract the entire residue into an array, convert the pucker and write the new residue in 
    #the new output list in the appropriate format
    newPdb=[]
    dealtWithResidueList=[]
    #extract all the atoms from the PDB (makes it much easier to find entire residues)
    atoms=extractAtomsFromPDB(pdb)

    #loop through the pdb file
    for line in pdb:
        if line[0:4]=='ATOM':
            #if we're dealing with an atom, extract the atom information
            atom = parsePdbLine(line)

            # is the atom in the residue list
            if atom[5] in puckerResiduesList:
                # has it been dealt with already; if so then do nothing.
                if not atom[5] in dealtWithResidueList:
                    try:
                        # create a list of all the atoms in the current residue in whatever order they appear in file
                        currentResidueAtoms=[curResAtom for curResAtom in atoms if curResAtom[5]==atom[5]]

                        # find out what state to convert the proline to
                        convertPuckerToState=[ convState[2] for convState in puckerState if convState[0]==atom[5]]
    
                        # generate the new list of residues
                        newResidue=convertPuckerState(currentResidueAtoms,convertPuckerToState)
    
                        # copy the new list to the output list
                        for atomNewResidue in newResidue:
                            newPdbLine=pdbLineFromAtom(atomNewResidue)
                            newPdb.append(newPdbLine)
    
                        # now it been dealt with then register residue as being done.
                        dealtWithResidueList.append(atom[5])
     
                    except:
                        print("unable to process atom: " )
                        print(atom )
                        exit(1)

            else:
                # if atom does not belong to a residue in the convert list copy it straight across.
                newPdb.append(line)
        else:
            # not an atom line so just copy across
            newPdb.append(line)    

    return newPdb

def readPuckerData(filename):
    puckerStateLines=readTextFile(filename)
    puckStateList=[]
    try:
        for residue in puckerStateLines:
            data=residue.split()
            # print(data,len(data) )
            if data[1]=='HYP' or data[1]=='PRO':
                puckStateList.append([int(data[0]),data[1],data[2]])
    except:
        print("Unable to read pucker file:" + filename)
        sys.exit(1)

    return puckStateList


def writePuckerData(outfile,pucker):
    try:
        vst = open(outfile, 'w')
    except:
        raise Exception( "Unable to open outfile: " + outfile )
    for puck in pucker:
        try:
            outStr=str(puck[0])+' '+puck[1]+' '+puck[2]+'\n'
        except:
            outStr=str(puck[0])+' '+puck[1]+'\n'

        # print(outStr)
        vst.write(outStr)
    
    vst.close()
    return

def fileRootFromInfile(infile):
    fileroot=infile
    if ".pdb" in infile:
        fileroot = infile.replace('.pdb','',1)
    return fileroot

def readpucker(infile, params):
    #unpack parameters
    outfile = params
    #perform required operations
    atoms=readAtoms( infile )
    pucker=checkPuckerList( atoms )
    writePuckerData( outfile, pucker )
    return

def writepucker(infile, params):
    #unpack parameters
    puckerfilename = params[0]
    outfile = params[1]

    #perform required operations
    pdb = readTextFile( infile )
    newPuckerState = readPuckerData( puckerfilename )
    newPdb = convertPdbPucker( pdb,newPuckerState )
    writeTextFile( newPdb,outfile )
    return


def removeLineList(infile, params):
    #unpack parameters
    rulefile = params[0]
    outfile = params[1]
 
    #perform required operations
    pdb=readTextFile(infile)
    #print("pdb")
    #print(pdb[0:3])
  
    itemsToRemove = readTextFile(rulefile)
    print("itemsToRemove" )
    print(itemsToRemove)
    newPDB = removeLines(pdb, itemsToRemove)
    writeTextFile(newPDB, outfile)
    return

# searches a list breaking when it find the first occurence of the substring in the list  
def findSubstringInList(the_list, substring, startAtEnd=False):
    retVal = -1 # assume failure
    
    if startAtEnd:
        # reverses the list and calls the function again to search from the beginning...
        retVal = findSubstringInList(reversed(the_list), substring, startAtEnd=False)
        if retVal>=0:
            retVal = len(the_list) - 1 - retVal
    else:
        # search the list breaking when we find the substring
        for idx, s in enumerate(the_list):
            if substring in s:
                # set the retVal to the index where we encounter the substring for the first time
                retVal = idx
                break        
    return retVal


def addTermini(infile, params):
    # Adds Termini to the beginning or end of a PDB
    # doesn't handle chains
    
    addN = False
    addC = False
    
    for param in params:
        if 'N' in param:
            addN = True
            try:
                NAlpha = float(param.split(',')[0][2:]) * np.pi/180.0
                NBeta = float(param.split(',')[1][0:]) * np.pi/180.0
            except:
                NAlpha = 0 * np.pi/180
                NBeta = 109 * np.pi/180
                
        if 'C' in param:
            addC = True
            try:
                CAlpha = float(param.split(',')[0][2:]) * np.pi/180.0
                CBeta = float(param.split(',')[1][0:]) * np.pi/180.0
            except:
                CAlpha = 0 * np.pi/180.0
                CBeta = 109 * np.pi/180.0
       
    if addN==addC==False:
        print("Trying to add Termini but no termini were specified: ", params)
        exit(1)
    
    # load the pdb info
    pdb = readTextFile(infile)
    
    # generate a list of atoms
    atoms = extractAtomsFromPDB(pdb)
    
    # generate a list of residues - interested in the first and last residues and their names, and chain
    residues = [ [atom[5], atom[3], atom[4]] for atom in atoms ]
    
    if addN: 
        if residues[0][1]=='ACE':
            print("N terminus already terminated with ACE")
        else:
            # generate the points to construct the TNB frame at end of chain
            for atom in atoms[0:30]:
                if (atom[5] == residues[0][0]): 
                    if atom[1]=='N':
                        NPos = np.array([atom[7], atom[8], atom[9]])
                    if atom[1]=='CA':                                        
                        CAPos = np.array([atom[7], atom[8], atom[9]])
                    if atom[1]=='C':                                        
                        CPos = np.array([atom[7], atom[8], atom[9]])

            # compute the position of the new terminus atom.
            bondlength = np.linalg.norm(NPos - CAPos)
            TNBFrame = constructTNBFrame(CPos, CAPos, NPos)
            ACEPos = NPos + bondlength * generateTNBVecXYZ(TNBFrame, NBeta, NAlpha) 

            # construct the information for the N terminus record
            NTerm = [' ']*15
            NTerm[0] = 0
            NTerm[1] = ' C'
            NTerm[2] = ' '
            NTerm[3] = 'ACE'
            NTerm[4] = residues[0][2]
            NTerm[5] = residues[0][0] - 1
            NTerm[6] = ' '
            NTerm[7] = ACEPos[0]
            NTerm[8] = ACEPos[1]
            NTerm[9] = ACEPos[2]
            NTerm[10] = 1.0
            NTerm[11] = 0.0
            NTerm[12] = '    '
            NTerm[13] = ' '
            NTerm[14] = ' '
            
            # convert record into a pdb string
            ACEString = pdbLineFromAtom(NTerm)
            
            # figure out where to insert the string
            indexOfFirstAtom = findSubstringInList(pdb, 'ATOM', startAtEnd=False)
            
            # insert the ACE String at the right point
            pdb.insert(indexOfFirstAtom, ACEString)
                   
            print("ACE added to N Terminus.")
                   
    if addC:
        if residues[-1][1]=='NME':
            print("C terminus already terminated with NME")
        else:
            for atom in atoms[-30:]:
                if (atom[5] == residues[-1][0]): 
                    if atom[1]=='N':
                        NPos = np.array([atom[7], atom[8], atom[9]])
                    if atom[1]=='CA':                                        
                        CAPos = np.array([atom[7], atom[8], atom[9]])
                    if atom[1]=='C':                                        
                        CPos = np.array([atom[7], atom[8], atom[9]])
    
            bondlength = np.linalg.norm(NPos - CAPos)
            TNBFrame = constructTNBFrame(NPos, CAPos, CPos)
            NMEPos = CPos + bondlength * generateTNBVecXYZ(TNBFrame, CBeta, CAlpha) 
            
            # find the correct index at which to place the new string
            indexOfLastAtom = findSubstringInList(pdb, 'ATOM', startAtEnd=True)
            lastAtomNum = int(parsePdbLine(pdb[indexOfLastAtom])[0])
                                    
            CTerm = [' ']*15
            CTerm[0] = lastAtomNum + 1 
            CTerm[1] = ' N'
            CTerm[2] = ' '
            CTerm[3] = 'NME'
            CTerm[4] = residues[-1][2]
            CTerm[5] = residues[-1][0] + 1
            CTerm[6] = ' '
            CTerm[7] = NMEPos[0]
            CTerm[8] = NMEPos[1]
            CTerm[9] = NMEPos[2]
            CTerm[10] = 1.0
            CTerm[11] = 0.0
            CTerm[12] = '    '
            CTerm[13] = ' '
            CTerm[14] = ' '
            
            # generate the NME string
            NMEString = pdbLineFromAtom(CTerm)
            
            
            # insert the ACE String at the right point (after the last atom hence + 1)
            pdb.insert(indexOfLastAtom + 1, NMEString)
            
            print("NME added to C Terminus.")
            
            # replace the terminal string with a simple ter
            indexOfTer = findSubstringInList(pdb, 'TER', startAtEnd=True)
            pdb[indexOfTer] = 'TER\n'
        
    # output the new file
    outfile = fileRootFromInfile(infile) + '_term.pdb'
    writeTextFile(pdb, outfile)

def fragmentPDB(infile, params):
    # assume only one chain
    pdb = readTextFile(infile)
    resData = readTextFile(params)
    fileRoot = fileRootFromInfile(infile)
    
    for resPair in resData:
        
        try:
            firstRes = int(resPair.split()[0])
            secondRes = int(resPair.split()[1])
            print("Residue Pair: ", firstRes, secondRes)
        except:
            print("Invalid res pair: ", resPair, " in file: ", params)
            exit(1)
        
        filename = fileRoot + '_' + str(firstRes).zfill(4) + '_' + str(secondRes).zfill(4) + '.pdb'

        outList = []

        for line in pdb:        

            # default is to output the line verbatim
            outputString = line
        
            # assuming we are keeping the line from the pdb
            skip = 0
        
            # split up each line into tokens.
            vals=line.split()
            
            # skip anisou lines
            if vals[0] in ['ANISOU']:
                skip=1

            # if we are dealing with an atom or hetatom
            if vals[0] in ['HETATM', 'ATOM']:
            
                # parse the pdb line
                atom = parsePdbLine(line)  
          
                # check to see if we are keeping this residue or not against the res pair limits 
                if int(atom[5]) < firstRes or int(atom[5]) > secondRes:  
                    skip = 1
            
            #if we are outputting this line then output it. 
            if skip==0:
                #write data to file
                outList.append(outputString)

        writeTextFile(outList, filename)
    
    return
    
        
def reThread(infile, newSequence):
    pdb = readTextFile(infile)
    
    # get the atoms
    atoms = extractAtomsFromPDB(pdb) 

    # extract the back bone    
    backbone = [ atom for atom in atoms if atom[1] in ['C', 'CA', 'N', 'O']]

    # generate the res names for each backbone atom in the new sequence
    newBackboneResNames = []
    for res in newSequence:
        newBackboneResNames.append(res)
        newBackboneResNames.append(res)
        newBackboneResNames.append(res)
        newBackboneResNames.append(res)

    # replace atom[5] with the new res name for all the atoms in the newResBackBone
    # renames len(newSequence) residues in the original pdb.
    for atom, resName in zip(backbone[0:len(newBackboneResNames)], newBackboneResNames):
        atom[5] = resName
    
    # save the pdb with the new atoms:
    

def centrePDB(inpFile):
    
    pdb = readTextFile(inpFile)
    
    # get the atoms
    atoms = extractAtomsFromPDB(pdb) 
    
    atomsXYZ = np.array( [ [float(atom[7]), float(atom[8]), float(atom[9])] for atom in atoms] )

    # compute centre of mass - technically should multiply by weight of atoms but not doing that. 
    # just finding geometric centre of blob.
    com = np.sum(atomsXYZ)/len(atomsXYZ)
    
    atomsXYZ = atomsXYZ - com
     
    replacePdbAtoms(inpFile, atomsXYZ, inpFile[0:-4] + '_com.pdb', pdb=pdb, atoms=atoms)
     
    

#this is hacked to do what I wanted it to do for one occasion and is not generalised
def symmetrize(forcefield, topologyPreSym, topology, pathname='~/svn/SCRIPTS/AMBER/symmetrise_prmtop/perm-prmtop.ff03'):
    command=[]
    if 'us' in forcefield:
        command = pathname + 'us.py ' + topologyPreSym + ' ' + topology
    else:
        command = pathname + '.py ' + topologyPreSym + ' ' + topology
    print(command)

    os.system(command)
    return


def renameTerminiTop(topology, pdbfile, addLetters):
    print(topology)
    print(pdbfile)
    print(addLetters)

    if addLetters==1:
        print('Adding Ns and Cs to Termini for symmetrization.\n')
    else:
        print('Removing Ns and Cs from Termini.\n')
    originalTopology=readTextFile(topology)
    pdbData=readTextFile(pdbfile)
    chainList=findChains(pdbData)

    finalTopology=[]

    #set logic control flags
    foundIt=0
    dealtWithIt=0

    #loop through each line in the output just copying and dumping verbatim in most cases.
    #simple state machine based on the flags
    #foundIt=0 dealtWithIt=0  just copying and dumping in the first part of file
    #foundIt=2 dealtWithIt=0  copying and dumping the format statement in the relevant part of the file
    #foundIt=1 dealtWithIt=0  outputting the renamed sequence correctly
    #foundIt=1 dealtWithIt=1  cycling through the sequence in the input file without doing anything
    #dealtWithIt=2  copying and dumping the second half of the file


    for line in originalTopology:
        newLine=[]

        #if we've found the right part, copied the two lines and have dealt with it, 
        # but not yet encountered the next section of the input file then do nothing
        #if we encounter the next section of the input file then copy it across
        #and log that we have completed the changes by setting dealtWithIt to 2
        if (foundIt==1)and(dealtWithIt==1):
            if line[0]=='%':
                dealtWithIt=2
                newLine=line
  

        #if we have found the right part of the file and copied the first two lines
        #but haven't yet dealt with the sequence information then do so.
        if (foundIt==1)and(dealtWithIt==0):
            dealtWithIt=1
            # build an outlist of residues in the correct sequence, adding Ns Cs or not as appropriate
            outList=[]
            # loop through each chain
            for chain in chainList:
                # keep track of where we are in the chain
                curRes=0
                # loop through the residues in the chain
                for residue in chain:
                    # create the output word
                    resWord=residue[1]+' '
                    # if we are in an N or C terminus situation and adding letters then do so.
                    if addLetters==1:
                        if curRes==0:
                            resWord='N'+residue[1]
                        if curRes==len(chain)-1:
                            resWord='C'+residue[1]
                    # print(resWord)
                    # add the current word to the list
                    outList.append(resWord)
                    # increment tracker
                    curRes=curRes+1

            # loop through list of residue words output strings of 20 in 4 char format. 
            curRes=0
            l = ''
            for res in outList:
                l = l+res
                curRes=curRes+1
             
            if curRes==20:
                finalTopology.append(l+'\n')
                l=''
                curRes=0

            # if there are unappended residues at end of for loop, output them.
            if l!='':
                finalTopology.append(l+'\n')

        # if we found the pertinent section on previous loop then copy format statement across
        if foundIt==2:
            newLine=line
            foundIt=foundIt-1
    
        # if we have found the pertinent part of the file then set a flag to say so
        if line[0:19]=='%FLAG RESIDUE_LABEL':
            newLine=line #copy the current line
            foundIt=2 #set the flag so on the next loop we copy one more line

        #if we haven't yet found the pertinent part on previous loops just copy the line across
        if (foundIt==0):
            newLine=line

        #if we've dealtwith it and found the next section of the file then just copy the line across.
        if dealtWithIt==2:
            newLine=line
 
        #if we created a line to copy then copy it
        if newLine!=[]:     
            finalTopology.append(newLine)

    #just over write the same file
    writeTextFile(finalTopology,topology)

    return

def findIndexVal(atoms, atomName, resNum):
    retVal = -1
    for atom in atoms:
        if (atom[1]==atomName) and (atom[5]==resNum):
            retVal = atom[0]
            break
    return retVal

def flipCT(infile, params):
    
    # Generic way to flip the cis trans that worked on one occasion is to do two rotations:
    # the O and C -90 about the N CA axis where the CA is in the same residue as the O and C 
    # and N and H -90 about the C and CA azis where the CA is in the same residue as the N and H
    cisTrans = readTextFile(params[0])
    forcefield = params[1]

    # read the atom information in from the PDB
    atoms = readAtoms(infile)

    # set up a pointer to the current input file    
    currentFile = cp.copy(infile)
    cleanup = []
    # loop through the lines of input in the cis trans state file
    for state, atomValStr in zip(cisTrans[1::2], cisTrans[0::2]):

        # only do anything if we encounter a Cis
        if 'C' in state:
            # atoms vals are the O C N H index in the PDB of the peptide bond always.
            atomVals = [ int(aVal) for aVal in atomValStr[0:-1].split() ]
            
            # Getthe index of the CA in same residue as the Oxygen (atomVals[0])
            resValOfTheO = atoms[atomVals[0] - 1][5]
            CAIndex = findIndexVal(atoms, 'CA', resValOfTheO)
            
            # Create atom group to perform first rotation of the O and C about the N CA axis:
            name = "group_" + str(resValOfTheO) + "_" + str(atomVals[0]) + "_1" 
            outLine1 = "GROUP " + name + " " + str(atomVals[2]) + " " + str(CAIndex) + " 2 -90\n"
            outLine2 = str(atomVals[0]) + "\n"
            outLine3 = str(atomVals[1])
            
            # write the atom group file for this first set of atoms
            writeTextFile([outLine1, outLine2, outLine3], name)
            
            # create the next file name
            nextFile = name + ".pdb"
            
            # do the rotation and save the results in nextFile.
            rotateGroup(currentFile, name, nextFile)

            # make a note of files to clean up
            cleanup.append(name)
            cleanup.append(currentFile)
            
            # make the new file the current file
            currentFile = cp.copy(nextFile)
            
            # Getthe index of the CA in same residue as the Nitrogen (atomVals[2])
            resValOfTheN = atoms[atomVals[2] - 1][5]
            CAIndex = findIndexVal(atoms, 'CA', resValOfTheN)
            
            # create the atom group for the second rotation of the N and H about the C CA axis:
            name = "group_" + str(resValOfTheO) + "_" + str(atomVals[0]) + "_2"
            outLine1 = "GROUP " + name + " " + str(atomVals[1]) + " " + str(CAIndex) + " 2 -90\n"
            outLine2 = str(atomVals[2]) + "\n"
            outLine3 = str(atomVals[3])
            
            # write the atom group file for the second set of atoms
            writeTextFile([outLine1, outLine2, outLine3], name)

            # create the next file name
            nextFile = name + ".pdb"
            
            # do the rotation and save the results in nextFile.
            rotateGroup(currentFile, name, nextFile)
            
            # make a note of files to clean up
            cleanup.append(name)
            cleanup.append(currentFile)
            
            # make the new file the current file
            currentFile = cp.copy(nextFile)

    # don't forget the last cheeky file
    cleanup.append(currentFile)
            
    # having performed all the rotations to flip the Cis/trans now PAG the final file with the given force field. 
    prepAmberGMin(currentFile, ['noreduce', forcefield])
    
    for fileN in cleanup:
        if fileN==infile:
            pass
        else:
            if ('leap' not in fileN):
                os.system("rm " + fileN)


def prepAmberGMin(infile, params, renameTermini=True):
    
    os.system("pwd")

    rulefile = params[0]
    forcefield = params[1]
    fileRoot = fileRootFromInfile(infile)
    cleanPDB = fileRoot + '_clean.pdb'
    topologyPreSym = fileRoot + '_preSym.prmtop'
    topology=fileRoot + '.prmtop'
    coords=fileRoot + '.inpcrd'
    outPDB=fileRoot + '_tleap.pdb'

    prepFileFlag=0
    paramsFlag=0

    #set flags and generate final keywords
    if len(params)>2:
        prepFile = params[2]
        prepFileFlag = 1
    if len(params)>3:
        paramsFile = params[3]
        paramsFlag = 1
 
    print(rulefile)
 
    if rulefile=='reduce':
        print("Reducing structure using reduce -Trim from Ambertools.")
        os.system("reduce " + infile + " -Trim > " + cleanPDB)
    else:
        if rulefile=='noreduce':
            print("Not removing or trimming any atoms.")
            os.system("cp " + infile + " " + cleanPDB)
        else:
            print("Removing atoms according to specified rule file.")
            removeLineList(infile, [rulefile, cleanPDB])

    #generate tleap input script
    vst=open("tleap.in",'w')
    vst.write("source leaprc." + forcefield + '\n')
    if prepFileFlag:
        vst.write("loadamberprep " + prepFile + '\n')
    if paramsFlag:
        vst.write("loadamberparams " + paramsFile + '\n')
    vst.write("mol=loadpdb " + cleanPDB + '\n')
    vst.write("saveamberparm mol " + topologyPreSym + " " + coords + '\n')
    vst.write("savepdb mol " + outPDB + '\n')
    vst.write("quit" + '\n')
    vst.close()
    os.system("tleap -f tleap.in")
    if renameTermini:
        renameTerminiTop(topologyPreSym, outPDB, 1)
        symmetrize(forcefield,topologyPreSym,topology)
        renameTerminiTop(topology, outPDB, 0)
    else:
        symmetrize(forcefield,topologyPreSym,topology)
    os.system("rm tleap.in")
    os.system("rm "+topologyPreSym)
    os.system("rm "+cleanPDB)
    os.system("mv "+topology+" coords.prmtop")
    os.system("mv "+coords+" coords.inpcrd")



    return

def renumberResidues(infile, startNum, outfile):

    #open input file
    fI=open(infile,'r')

    #read in the data
    rawData=fI.readlines()

    #close input file
    fI.close()
 
    #open the outputfile
    fO=open(outfile,'w')

    #set starting counters
    curOutputResId=int(startNum)
    for line in rawData:
        if line.split()[0] in ['HETATM', 'ATOM']:
            oldResId=line.split()[5]
            curResName=line.split()[3]
            break
        
    atomNumber=1
    curChainIndex=0
    chainList='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    curChain=chainList[curChainIndex]
   
    #loop through the input data one line at a time
    for line in rawData:
        
        #default is to output the line verbatim
        outputString=line
        
        #split up each line into tokens.
        vals=line.split()
        
        #set up control flag
        skip=0
        
        #check to see if there is an ANISOU line present
        if vals[0] in ['ANISOU']:
            skip=1

        #if we are dealing with an atom or hetatom
        if vals[0] in ['HETATM', 'ATOM']:
            
            #parse the pdb line
            atom=parsePdbLine(line)  
          
            #update the atom number
            atom[0]=atomNumber
            atomNumber+=1
          
            #if the test residue ID is different from last time then increment the residue number
            if int(atom[5])!=int(oldResId):
                oldResId=atom[5]
                curResName=atom[3]
                curOutputResId=curOutputResId+1
            
            #output the curOutputResId for this residue
            atom[5]=curOutputResId
                
            #set the chain value
            atom[4]=curChain
            
            #generate the new string
            outputString=pdbLineFromAtom(atom)
            
        if vals[0] in ['TER']:
            outputString='TER  {: >06d}      {:3} {:1}{: >4d}\n'.format(atomNumber, curResName, curChain, curOutputResId)
            atomNumber+=1
            curChainIndex+=1
            curChain=chainList[curChainIndex]

        #if outputstring is not 'skip' then outputString to file
        if skip==0:
            #write data to file
            fO.write(outputString)

    #when we've reached the end of the input data, close the output
    fO.close()
    
    return

def sortResidues(infile,sortfile,outfile):

    rawData=readTextFile(infile)
    atoms=readAtoms(infile)
    sortData=readTextFile(sortfile)
    #trim off carriage returns - might be operating system dependent. works today!
    sortData=[sort[0:-1] for sort in sortData]

    #open the outputfile
    fO=open(outfile,'w')

    #output the file header until we hit the first atom line
    for line in rawData:
        #split the current line into tokens
        tokens=line.split()

        #if we encounter an atom then break the loop
        if tokens[0] in ['ATOM','HETATM']:
            break

        #write the input line to file
        fO.write(line)

    #initialise loop variables
    chains=' ABCDEDGHIJKLMNOPQRSTUVWXYZ'
    curChainIndex=0

    #start the current residue output counter (avoids having to do a renumber later on)
    curRes=0

    #set the atom counter
    curAtom=0

    #Initialise Resname
    resName='UNK'

    #now just working on outputting atoms and chains in the right order. Search the input data for the relevant residue information
    #and output according to the instructions in the sort file. The sort file contains the order of the residue numbers as numbered in the
    #original information
    for sortRes in sortData:
  
        #check for keywords in the sort file
        if sortRes in ['CHAIN','TER', '']:
            # if it's the start of a new chain then increment the chain letter
            if sortRes=='CHAIN':
                curChainIndex=curChainIndex+1
                curChain=chains[curChainIndex]
   
            #if it's the end of a chain then output a TER command
            if sortRes=='TER':
                curAtom=curAtom+1
                l='TER  {: >06d}      {:3} {:1}{: >4d}\n'.format(curAtom, resName, curChain, curRes)
                fO.write(l)
               
        else:
            #otherwise sortRes is the residue in the input data file. Increment the curRes (initialised to zero)
            curRes=curRes+1
  
            #find all the atom data for that residue
            curResidues=[atom for atom in atoms if atom[5]==int(sortRes)]

            #get the resName for setting the TER statements correctly
            resName=curResidues[0][3]

            #loop through the atoms in the current residue - need to renumber the atoms, residue and chains.
            for atom in curResidues:
                newAtom=atom[:]
                #increment the atom counter (never gets reset)
                curAtom=curAtom+1
                #rewrite the data in the new atom (leave the original data intact)
                newAtom[0]=curAtom
                newAtom[4]=curChain
                newAtom[5]=curRes
 
                #convert new atom to string and output to file
                fO.write(pdbLineFromAtom(newAtom))

    #terminate the PDB file when all is said and done
    fO.write('END')

    #close the output file
    fO.close()

    return

def proToHyp(infile,convFile,outFile):

    rawData=readTextFile(infile)
    atoms=readAtoms(infile)
    convData=readTextFile(convFile)

    #trim off carriage returns - might be operating system dependent. works today!
    convData=[ conv[0:-1] for conv in convData]

    #open the outputfile
    fO=open(outFile,'w')

    #maintain a counter for which atom we are looking at.
    curAtom=-1
 
    #copy across the raw data verbatim unless we hit and atomline
    for line in rawData:
        #split the current line into tokens
        tokens=line.split()

        #if we encounter an atom then decide if we need to modify it
        if tokens[0] in ['ATOM','HETATM']:

            #get the next atom from the list (should tally up with the raw data list. fingers crossed).
            curAtom+=1

            #extract the atom
            atom=atoms[curAtom]

            #if the residue is to be converted then convert the atom.
            if str(atom[5]) in convData:
                atom[3]='HYP'
                if atom[1]=='HG3':
                    atom[1]='OD1'
            l=pdbLineFromAtom(atom)
            fO.write(l)
        else:
            #write the input line to file without mods
            fO.write(line)

    #close file when there is no more input data
    fO.close()

    return

def convertSequence(infile, outfile):
    singleLetterSequence = readTextFile(infile)
    print(singleLetterSequence)
    outSequence = []
    for letter in singleLetterSequence[0]:
        if letter=='A':
            outSequence.append('ALA\n')
        if letter=='R':
            outSequence.append('ARG\n')
        if letter=='N':
            outSequence.append('ASN\n')
        if letter=='D':
            outSequence.append('ASP\n')
        if letter=='C':
            outSequence.append('CYS\n')
        if letter=='Q':
            outSequence.append('GLN\n')
        if letter=='G':
            outSequence.append('GLY\n')
        if letter=='E':
            outSequence.append('GLU\n')
        if letter=='H':
            outSequence.append('HIS\n')
        if letter=='I':
            outSequence.append('ILE\n')
        if letter=='L':
            outSequence.append('LEU\n')
        if letter=='K':
            outSequence.append('LYS\n')
        if letter=='M':
            outSequence.append('MET\n')
        if letter=='F':
            outSequence.append('PHE\n')
        if letter=='P':
            outSequence.append('PRO\n')
        if letter=='S':
            outSequence.append('SER\n')
        if letter=='T':
            outSequence.append('THR\n')
        if letter=='W':
            outSequence.append('TRP\n')
        if letter=='Y':
            outSequence.append('TYR\n')
        if letter=='V':
            outSequence.append('VAL\n')
    writeTextFile(outSequence, outfile)
        

def readSequence(infile, mode, outfile, width=80):
    atoms=readAtoms(infile)
    residues=findResidues(atoms)
    
    #mode 1 numbered with three letter codes on separate lines
    if mode==1:
        print("mode: 1 selected")
        writeTextFile([str(res[0])+' '+str(res[1])+'\n' for res in residues],outfile)
    #mode 1 unumbered with three letter codes on separate lines - suitable for a modify sequence command
    if mode==2:
        print("mode: 2 selected")
        writeTextFile([str(res[1])+'\n' for res in residues],outfile)
    #mode 3 string of first letters only
    if mode==3:
        print("mode: 3 selected")
        l=''
        for res in residues:
            l += getSingleLetterFromThreeLetterAACode(res[1])
        l+='\n'
        writeTextFile(l,outfile)
    
    # 3 letter sequence as a single line
    if mode==4:
        print("mode: 4 selected")
        writeTextFile([str(res[1])+' ' for res in residues],outfile)
    
    if mode==5:
        print("mode: 5 selected (fixed width format)")
        l=''
        count = 0
        for res in residues:
            l += getSingleLetterFromThreeLetterAACode(res[1])
            count +=1
            if count==width:
                count = 0
                l += '\n'
        if not count==width:
            l += '\n'
        writeTextFile(l,outfile)
        
    return

def getSingleLetterFromThreeLetterAACode(res):
    # copy first letter in cases where that works
    if not res in ['ARG','LYS','ASP','GLU','ASN','GLN','SEC','PHE','TYR','TRP']:
        l = res[0]
    else:
        # specify explicitly where first letter is not the code
        if res=='ARG':
            l = 'R'
        if res=='LYS':
            l = 'K'
        if res=='ASP':
            l = 'D'
        if res=='GLU':
            l = 'E'
        if res=='ASN':
            l = 'N'        
        if res=='GLN':
            l = 'Q'        
        if res=='SEC':
            l = 'U'
        if res=='PHE':
            l = 'F'                
        if res=='TYR':
            l = 'Y'
        if res=='TRP':
            l = 'W'

    return l
    
def modifySequence(infile, newSequence, startResidue, outFile):
    
    rawData=readTextFile(infile)
    atoms=readAtoms(infile)
    newSeq=readTextFile(newSequence)
    startResidue=int(startResidue)

    #trim off carriage returns - might be operating system dependent. works today!
    newSeq=[ newS[0:-1] for newS in newSeq]
    
    #open the outputfile
    fO=open(outFile,'w')

    #There are two list of atoms: an input (atoms) and an output (fO).
    curAtomInputIndex=0
    curAtomOutputIndex=1
 
    #Create an index for the replacement sequence (newSeq) 
    newResSeqIndex=0  #always zero based for first entry

    #count the number of existing residues we have already processed. Initialise to zero.
    resCount=0

    #keep note of number of the last residue that we processed
    lastResNumber=atoms[0][5] #initialise to the residue number of the first atom in the input list

    #create a variable to store the name of the last residue processed
    lastResName=atoms[0][3] #initialise to the residue name of the first atom in the input list
    
    #initialy we are not renaming until a set number of residues has been processed 
    renaming=False
    
    #copy across the raw data verbatim and dump to file.
    #if we hit an ATOM or HETATM decide whether or not to output the atom.
        #if we output the atom then replace the residue name and atom number 
    #if we hit a TER we must output with the name of the last redisue and increment the atomnumber    
    for line in rawData:
        #split the current line into tokens
        tokens=line.split()
        
        #if we encounter an atom line then decide if we need to modify it or ignore it.
        if tokens[0] in ['ATOM','HETATM']:
  
            #extract the data for the current atom
            atom=atoms[curAtomInputIndex]
            
            #increment the input index to the next atom for the next cycle
            curAtomInputIndex+=1

            #if the residue number has changed from last cycle then we have finished processing the previous residue.
            #Increment the number of residues processed.
            if atom[5]!=lastResNumber:
                #make a note of the new res number and name
                lastResNumber=atom[5]
                lastResName=atom[3]
                
                #increment the number of residues processed
                resCount+=1
                
                #if we are renaming then also increment the new sequence residue number
                if renaming:
                    newResSeqIndex+=1
                    if newResSeqIndex>=len(newSeq):
                        renaming=False
                    else:
                        lastResName=newSeq[newResSeqIndex]                    

            #check the conditions for jumping into renaming mode.        
            if ((resCount>=startResidue-1) and (newResSeqIndex<len(newSeq))):
                renaming=True

            #debug statement
            #print(atom[5], newResSeqIndex, resCount, lastResNumber, lastResProc)

            #Do we need to reprocess the current residue?
            if (renaming):
                #Is the new sequence name different from the existing sequence?
                if atom[3]!=newSeq[newResSeqIndex]:
                    #Only output atoms if they are backbone atoms, or belongs to an NME or ACE residue
                    if (str(atom[1]) in ['C','N','O','H','CA']) or (atom[3] in ['NME','ACE']):
                        #give the sequence a new name
                        atom[3]=newSeq[newResSeqIndex]
                        
                        #give the atom a new number
                        atom[0]=int(curAtomOutputIndex)
                        
                        #output the atom to file
                        fO.write(pdbLineFromAtom(atom))
                        
                        #increment the atom output index
                        curAtomOutputIndex+=1
                else:
                    #new sequence is the same as the old one so just output the atom as is but with a new atom number
                    #give the atom a new number
                    atom[0]=int(curAtomOutputIndex)
                    #output the atom to file
                    fO.write(pdbLineFromAtom(atom))
                    #increment the atom output index
                    curAtomOutputIndex+=1
            else:
                #We are not renaming the current residue so just output the atom with a new number
                atom[0]=int(curAtomOutputIndex)
                #output the line
                fO.write(pdbLineFromAtom(atom))
                #increment the atom output index
                curAtomOutputIndex+=1
        else:
            if tokens[0] in ['TER']:
                l='TER' #  {: >06d}      {:3} {:1}{: >4d}\n'.format(curAtomOutputIndex, lastResName, tokens[3], int(tokens[4]))
                fO.write(l)
                curAtomOutputIndex+=1       
            else:
                #write the input line to file without mods
                fO.write(line)

    #close file when there is no more input data
    fO.close()

    return


def puckerGroupsRB(infile,outfile):
    '''Function reads the pdb file from infile and outputs all the HYP and PRO residues as groups in the rbodyconfig file'''
    #get the atoms from the infile
    atoms=readAtoms(infile)
 
    #open the outputfile
    fO=open(outfile,'w')

    #generate a residue list
    resList=findResidues(atoms)

    #loop through the residues
    for res in resList:
        if res[1] in ['HYP','PRO']:
            #extract the atoms for this residue
            atomList=getAtomsInResidueAll(atoms,res[0])

            #write the next line of the output file
            fO.write('GROUP '+str(len(atomList))+'\n')

            #output the atom numbers
            for atom in atomList:
                fO.write(str(int(atom[0]))+'\n')

    fO.close()
    
    return

def puckerGroupsRBs(infile,resfile,outfile):
    '''Function reads the pdb file from infile and outputs the HYP and PRO residues as groups in the rbodyconfig file, if they are specified in the resfile'''
    #get the atoms from the infile
    atoms=readAtoms(infile)

    #get the list of residues of interest from the resfile
    resList=readTextFile(resfile)
    
    #open the outputfile
    fO=open(outfile,'w')

    for resnumberRaw in resList:
        resnumber=resnumberRaw[:-1]

        #extract atoms
        atomList=getAtomsInResidueAll(atoms,int(resnumber))

        #check this is a sensible residue for outputting
        if atomList[0][3] in ['HYP','PRO']:

            #write the next line of the output file
            fO.write('GROUP '+str(len(atomList))+'\n')

            #output the atom numbers
            for atom in atomList:
                fO.write(str(int(atom[0]))+'\n')
        else:
            print('residue is not HYP or PRO.')
 
    fO.close()

    return

def puckerGroups(infile,OHFlag, outfile):

    print('OHFlag: ',OHFlag)

    #rawData = readTextFile(infile)
    atoms = readAtoms(infile)
 
    #open the outputfile
    fO=open(outfile,'w')

    #generate the residue list
    resList=findResidues(atoms)

    fO1=open('resList','w')

    for res in resList:
        if res[1] in ['HYP','PRO']:
            #output the selected residue
            fO1.write(str(res[0])+'\n')

            #create a name for the group
            name=str(res[1])+'_'+str(res[0])
            nameOH=str(res[1])+'_'+str(res[0])+'_OH'

            #extract atoms
            atomList=getAtomsInResidueAll(atoms,res[0])

            axisAtoms=[]
            groupAtoms=[]
            axisAtomsOH=[]
            groupAtomsOH=[]
            #loop throught list of atoms in the residue extracting the appropriate values
            for atom in atomList:
                #CD to CB determines the axis of rotation
                if atom[1] in ['CD','CB']:
                    axisAtoms.append(int(atom[0]))
                if atom[1] in ['CG','OD1']:
                    axisAtomsOH.append(int(atom[0]))
                if atom[1] in ['HD2','HD3','CG','HG2','HG3','HB2','HB3','OD1','HO1']:
                    groupAtoms.append(int(atom[0]))
                if atom[1] in ['HO1']:
                    groupAtomsOH.append(int(atom[0]))
        
            #write the next line of the output file
            fO.write('GROUP '+name+' '+str(axisAtoms[0])+' '+str(axisAtoms[1])+' '+str(len(groupAtoms))+' 1.0 1.0\n')
            #output the atoms numbers
            for atom in groupAtoms:
                fO.write(str(atom)+'\n')

            #output the OH group rotation if in a HYP and OHFlag is set
            if (OHFlag==str(1)) and (res[1]=='HYP'):
                #write the next line of the output file
                fO.write('GROUP '+nameOH+' '+str(axisAtomsOH[0])+' '+str(axisAtomsOH[1])+' '+str(len(groupAtomsOH))+' 1.0 1.0\n')
                #output the atoms numbers
                for atom in groupAtomsOH:
                    fO.write(str(atom)+'\n')

    fO1.close()
    fO.close()
    return

def puckerGroupSpec(infile, resfile, OHFlag, scaleFac, rotProb, outfile):
   
    #rawData=readTextFile(infile)
    resList=readTextFile(resfile)
    atoms=readAtoms(infile)

    #open the outputfile
    fO=open(outfile,'w')

    for resnumberRaw in resList:
        resnumber=resnumberRaw[:-1]

        #extract atoms
        atomList=getAtomsInResidueAll(atoms,int(resnumber))
    
        #extract residuename
        resName=atomList[0][3]
    
        #check this is sensible
        if resName in ['HYP','PRO']:
            #create a name for the group
            name=str(resName)+'_'+str(resnumber)
            nameOH=str(resName)+'_'+str(resnumber)+'_OH'
   
            axisAtoms=[]
            groupAtoms=[]
            axisAtomsOH=[]
            groupAtomsOH=[]
            #loop throught list of atoms in the residue extracting the appropriate values
            for atom in atomList:
                if atom[1] in ['CD','CB']:
                    axisAtoms.append(int(atom[0]))
                if atom[1] in ['CG','OD1']:
                    axisAtomsOH.append(int(atom[0]))
                if atom[1] in ['HD2','HD3','CG','HG2','HG3','HB2','HB3','OD1','HO1']:
                    groupAtoms.append(int(atom[0]))
                if atom[1] in ['HO1']:
                    groupAtomsOH.append(int(atom[0]))

           
            #write the next line of the output file
            fO.write('GROUP '+name+' '+str(axisAtoms[0])+' '+str(axisAtoms[1])+' '+str(len(groupAtoms))+' '+str(scaleFac)+' '+str(rotProb)+'\n')
    
            #output the atoms numbers
            for atom in groupAtoms:
                fO.write(str(atom)+'\n')

            #output the OH group rotation if in a HYP and OHFlag is set
            if (int(OHFlag)==1) and (resName=='HYP'):
                #write the next line of the output file
                fO.write('GROUP '+nameOH+' '+str(axisAtomsOH[0])+' '+str(axisAtomsOH[1])+' '+str(len(groupAtomsOH))+' 1.0 '+str(rotProb)+'\n')
                #output the atoms numbers
                for atom in groupAtomsOH:
                    fO.write(str(atom)+'\n')

        else:
            print('residue is not HYP or PRO.')


    fO.close()
    return


def cpoa(p1List,p2List,p3List,p4List):
    '''Computes the point of closest approach between the vector between p1 and p2 and the vector between p3 and p4.
    p1List, p2List, p3List, and p4List are lists of numpy arrays of the same length
    
    Returns Lists of: 
    pa - the vector on the line p2-p1 which is at the point of closest approach
    mua - distance from p1 to pa.
    pb - the vector on the line p4-p3 which is at the point of closest approach
    mub - distance from p1 to pb.'''
    p13List=[p1-p3 for p1,p3 in zip(p1List,p3List)]
    p43List=[(p4-p3)/np.linalg.norm(p4-p3) for p4,p3 in zip(p4List,p3List)]
    p21List=[(p2-p1)/np.linalg.norm(p2-p1) for p2,p1 in zip(p2List,p1List)]
    
    d1343List=[np.dot(p13,p43) for p13,p43 in zip(p13List,p43List)]
    d4321List=[np.dot(p43,p21) for p43,p21 in zip(p43List,p21List)]
    d1321List=[np.dot(p13,p21) for p13,p21 in zip(p13List,p21List)]
    d4343List=[np.dot(p43,p43) for p43 in p43List]
    d2121List=[np.dot(p21,p21) for p21 in p21List]

    denomList=[(d2121*d4343) - (d4321 * d4321) for d2121,d4343,d4321 in zip(d2121List,d4343List,d4321List)]
    numerList=[(d1343*d4321) - (d1321 * d4343) for d1343,d4321,d1321,d4343 in zip(d1343List,d4321List,d1321List,d4343List)]

    mubList=[]
    muaList=[]
    pbList=[]
    paList=[]
    for denom,numer,d1343,d4321,d4343,p1,p21,p3,p43 in zip(denomList,numerList,d1343List, d4321List,d4343List,p1List,p21List,p3List,p43List):
        mua=0
        mub=0
        pa=p1
        pb=p3
        if (abs(denom)>1e-14):
            mua=numer/denom
            mub= (d1343+ d4321*(numer/denom))/d4343
            pa=p1+mua*p21
            pb=p3+mub*p43
    
        muaList.append(mua)
        mubList.append(mub)
        paList.append(pa)
        pbList.append(pb)
        
    return [muaList,paList,mubList,pbList]


def plotVectors(fig,listA,listB,listCol,blockFlag):
    '''Function plots an arrow between each vector in list a and each corresponding vector in list b'''
    maxVals=np.amax(listA+listB,0)
    minVals=np.amin(listA+listB,0)

    #draw a new axis if required
    axList=fig.get_axes()
    if not axList:
        ax = fig.gca(projection='3d')
    else:
        ax=axList[0]
    
    ax.clear()
    ax.set_aspect("auto")
    ax.set_xlim3d(minVals[0],maxVals[0])
    ax.set_ylim3d(minVals[1],maxVals[1])
    ax.set_zlim3d(minVals[2],maxVals[2])
                
    arrowList =[ Arrow3D([a[0],b[0]],[a[1],b[1]],[a[2],b[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color=col) for a,b,col in zip(listA,listB,listCol)]
    for a in arrowList:
        ax.add_artist(a)

    plt.draw()
    #plt.show(block=False)
    plt.show(block=blockFlag)

    return
    
def atomsToCOM(atoms):
    '''sets an atoms array to its centre of mass. Returns the COM and the new array'''
    newAtoms=[]
    COM=np.array([0.0,0.0,0.0])
    for atom in atoms:
        COM+=np.array([atom[7],atom[8],atom[9]])
    COM/=len(atoms)
    for atom in atoms:
        newAtom=[item for item in atom]
        newAtom[7]-=COM[0]
        newAtom[8]-=COM[1]
        newAtom[9]-=COM[2]
        newAtoms.append(newAtom)
        
    return [COM,newAtoms]


def readResidueSymmetry(atoms):
    '''Function takes atomic coords and analyses them to determine the helical params of a structure assuming it is collagen.
        Uses same methodology as readSymmetryFromAtoms and readUnitSymmetry.  
        The results are returned as an array of helical values for each residue based on the distance along the axis
        and twist about the axis to the equivalent residue in the GXY repeat. 
        '''
    
    #always ignore four residues at either end of chain. Allows the +1 residues to generate difference etc.
    capRes=4
    
    #set up output arrays
    AllAtomList=[]
    ResGList=[]
    ResXList=[]
    ResYList=[]
    CAGList=[]
    CAXList=[]
    CAYList=[]    
    
    #first move the atoms to the COM frame. Do everything there.
    [COM, newAtoms] = atomsToCOM(atoms)
    
    #identify the chains in the atoms
    chains=breakAtomsInToChains(newAtoms)
    
    #generate a list of residues in each chain
    residuesInChains=[ findResidues(chain) for chain in chains]
    
    #generate a GXY letter for each residue
    GXYList =[ idGXYPatternAll(chain) for chain in residuesInChains]
    
    #group the atoms in each residues into sublists
    residueList=[ breakChainIntoResidues(chain) for chain in chains ]
    
    #Extract arrays of useful atoms coords:
    #    the back bone of each chain (N,CA,C)
    #    The CAs of each Gly, Pro (X) and Hyp (Y)
        
    for residues,GXYInfo in zip(residueList,GXYList):
        
        #create arrays to store the coords of the atoms we are interested in in the current residue
        atomicCoords=[]
        CAG=[]
        ResG=[]
        CAX=[]
        ResX=[]        
        CAY=[]
        ResY=[]    

        #Get the back bone atoms from the residues that are not within capRes residues of either end.
        for curRes, GXY in zip(residues[capRes:-capRes:1],GXYInfo[capRes:-capRes-3:1]):
            #extract arrays of atoms of interest
            for atom in curRes:
                if atom[1] in ['N','C','CA']:
                    atomicCoords.append(np.array([atom[7],atom[8],atom[9]]))  


        #Ignore first GXY repeat. Include the last one. get the Ca from gly, pro and hyp residues.
        for curRes, GXY in zip(residues[capRes:-capRes:1],GXYInfo[capRes:-(capRes-3):1]):
            #extract arrays of atoms of interest
            for atom in curRes:
                if atom[1] in ['CA']:
                    if GXY in ['G']:
                        CAG.append(np.array([atom[7],atom[8],atom[9]]))
                        ResG.append(atom[5])
                    if GXY in ['X']:
                        CAX.append(np.array([atom[7],atom[8],atom[9]]))
                        ResX.append(atom[5])
                    if GXY in ['Y']:
                        CAY.append(np.array([atom[7],atom[8],atom[9]]))
                        ResY.append(atom[5])
                        
        #remember each chain of atomic coords separately
        AllAtomList.append(atomicCoords) #a list of positions of all the Ns, Cs and Ca throw away the id information. Just a set of points.
        ResGList.append(ResG)
        ResXList.append(ResX)
        ResYList.append(ResY)
        CAGList.append(CAG)
        CAXList.append(CAX)
        CAYList.append(CAY)

    #Take the mean of the first atom in each chain to be the base of the helix
    basePoint3Space=np.mean([AllAtomList[0][0],AllAtomList[1][0],AllAtomList[2][0]],0)

    #Take the radius guess as the average of the distances from the basePoint3Space to the start of each chain
    radiusGuess=np.mean([np.linalg.norm(AllAtomList[0][0]-basePoint3Space),np.linalg.norm(AllAtomList[1][0]-basePoint3Space),np.linalg.norm(AllAtomList[2][0]-basePoint3Space)])

    #Take an initial guess at the axis as the vector between basepoint and the centre of mass: COM=([0,0,0])
    zVecGuess=(np.array([0.0,0.0,0.0])-basePoint3Space)/np.linalg.norm(basePoint3Space) 

    #compute point at z=0 for line through basepoint in direction of zVecGuess. The length of the cylinder is irrelevant. Thus we can use the point where the axis crosses the yx plane to define the base point.
    #This eliminates another parameter for this fit. Only a problem if the z-axis is parallel to the xy plane. we'll cross that bridge when it turns up. 
    #basePoint2Space=array([-1.0*ZVecMeanNorm[0]*basePoint3Space[2]/ZVecMeanNorm[2]+basePoint3Space[0],-1.0*ZVecMeanNorm[1]*basePoint3Space[2]/ZVecMeanNorm[2]+basePoint3Space[1],0])
    basePoint2Space=np.array([-1.0*zVecGuess[0]*basePoint3Space[2]/zVecGuess[2]+basePoint3Space[0],-1.0*zVecGuess[1]*basePoint3Space[2]/zVecGuess[2]+basePoint3Space[1],0])
    
    #compute the polar coords of the Z-vector - only need the orientation as this is always a normal vector - reduces params by one.
    #ignore rZHat
    [rZHat,phiZHat,thetaZHat]=XYZTo3DPolar(zVecGuess)
    
    #Now have estimates for ZVector, the basepoint and the radius which defines a cylinder. Set it up as an array of parameters    
    InitialGuess=[radiusGuess,thetaZHat,phiZHat,basePoint2Space[0],basePoint2Space[1]]

    #flatten the atom list so we just have a single list of the all the back bone atoms vectors.
    AllAtomsListFlat=[item for sublist in AllAtomList for item in sublist]

    #find the overall range of the data vectors
    maxVals=np.amax(AllAtomsListFlat,0)
    minVals=np.amin(AllAtomsListFlat,0)
    
    #the radius ought not to be larger than half the longest possible distance within the cloud of data and must be greater than 0. whacked in a 20 because collagen is long and thin and fitting was being silly.
    radiusRange=np.linalg.norm(maxVals-minVals)/10
    
    #The base point parameter on the xy plane should not move more than radiusRange in any direction from the initial basepoint guess 
    
    #Set the bounds; the phi and theta bounds are set to be a more than pi to allow wrap arounds. It tended to get stuck at pi and not know that it could move beyond... seems to work...
    #If the gradient is going up when a parameter hits the boundary it stays there. If it can go past the boundary and the gradient becomes -ve it jumps away from the boundary...
    bounds=[(0,radiusRange),(0,2*np.pi),(-2*np.pi,2*np.pi),(basePoint2Space[0]-radiusRange,basePoint2Space[0]+radiusRange),(basePoint2Space[1]-radiusRange,basePoint2Space[1]+radiusRange)]
    
    #create a function pointer - [] is for plotting out stuff on the fly to monitor progress if desired
    f=(lambda x: radialDistanceProjectionSum(x, AllAtomsListFlat, []))
    
    #minimise the RadialDistanceProjection sum starting with the initial guess
    finalFit = fmin_l_bfgs_b(f, InitialGuess, approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0)
    #print('initial Z Vector Guess: ', zVecGuess)
    #print('Initial params guess (radius,theta,phi,basePointX,basePointY): ', InitialGuess)
    #print('Final params and errors:', finalFit)
    
    #construct the final vectors for the helix axis.
    zFinal=ThreeDPolarToXYZ(1.0,float(finalFit[0][1]),float(finalFit[0][2]))
    basePointFinal=np.array([finalFit[0][3],finalFit[0][4],0])

    #Now we have the z-axis vector and a defined point on the axis we can compute interesting things

    #generate output arrays
    ThetaCAG=[]
    ThetaCAX=[]
    ThetaCAY=[]
    DMagsCAG=[]
    DMagsCAX=[]
    DMagsCAY=[]
    RadiusCAG=[]
    RadiusCAX=[]
    RadiusCAY=[]

    #We previously extract four reference points from each GXY repeating unit
    # The N of Gly and the CAlphas for Gly, X and Y positions. 
    # Can use to analyse the rotation and spacing along the structure. 
    # Look at each chain separately.
    #compute the RadiusCAG, CAX and CAY for each repeat unit in each chain        
    for CAG,CAX,CAY in zip(CAGList,CAXList,CAYList):
        #compute position of each point relative to final basePoint
        basePointCoordsCAG= [point-basePointFinal for point in CAG]
        basePointCoordsCAX= [point-basePointFinal for point in CAX]
        basePointCoordsCAY= [point-basePointFinal for point in CAY]
        
        #project the data onto the zAxis
        zComponentsCAG=[np.dot(point,zFinal) for point in basePointCoordsCAG]
        zComponentsCAX=[np.dot(point,zFinal) for point in basePointCoordsCAX]
        zComponentsCAY=[np.dot(point,zFinal) for point in basePointCoordsCAY]
        
        #project the point onto the base plane
        basePlaneVectorCAG=[ point - zComp*zFinal for point, zComp in zip(basePointCoordsCAG,zComponentsCAG)]        
        basePlaneVectorCAX=[ point - zComp*zFinal for point, zComp in zip(basePointCoordsCAX,zComponentsCAX)]
        basePlaneVectorCAY=[ point - zComp*zFinal for point, zComp in zip(basePointCoordsCAY,zComponentsCAY)]        
        
        #Theta is the angle between each adjacent base plane Vectors in the same strand; return in degrees
        ThetaCAG.append([np.arccos(np.dot(bPlaneVec/np.linalg.norm(bPlaneVec),basePlaneVectorCAG[bvCurIndex+1]/np.linalg.norm(basePlaneVectorCAG[bvCurIndex+1])))*180.0/np.pi for bvCurIndex, bPlaneVec in enumerate(basePlaneVectorCAG[:-1])])
        ThetaCAX.append([np.arccos(np.dot(bPlaneVec/np.linalg.norm(bPlaneVec),basePlaneVectorCAX[bvCurIndex+1]/np.linalg.norm(basePlaneVectorCAX[bvCurIndex+1])))*180.0/np.pi for bvCurIndex, bPlaneVec in enumerate(basePlaneVectorCAX[:-1])])        
        ThetaCAY.append([np.arccos(np.dot(bPlaneVec/np.linalg.norm(bPlaneVec),basePlaneVectorCAY[bvCurIndex+1]/np.linalg.norm(basePlaneVectorCAY[bvCurIndex+1])))*180.0/np.pi for bvCurIndex, bPlaneVec in enumerate(basePlaneVectorCAY[:-1])])        

        #Vertical displacement between units is the difference between each ZComponent
        DMagsCAG.append([abs(zComponentsCAG[zCurIndex+1]-curZ) for zCurIndex, curZ in enumerate(zComponentsCAG[:-1])])
        DMagsCAX.append([abs(zComponentsCAX[zCurIndex+1]-curZ) for zCurIndex, curZ in enumerate(zComponentsCAX[:-1])])
        DMagsCAY.append([abs(zComponentsCAY[zCurIndex+1]-curZ) for zCurIndex, curZ in enumerate(zComponentsCAY[:-1])])        
        
        #Radius of the different CAlphas given by magnitude of the basePlaneVectors
        RadiusCAG.append([np.linalg.norm(bpvCAG) for bpvCAG in basePlaneVectorCAG])
        RadiusCAX.append([np.linalg.norm(bpvCAX) for bpvCAX in basePlaneVectorCAX])
        RadiusCAY.append([np.linalg.norm(bpvCAY) for bpvCAY in basePlaneVectorCAY])
        
    #collapse the outermost chain structure for all the output arrays
    ResG=[item for sublist in ResGList for item in sublist]
    ResX=[item for sublist in ResXList for item in sublist]
    ResY=[item for sublist in ResYList for item in sublist]
    ThetaCAG=[item for sublist in ThetaCAG for item in sublist]
    ThetaCAX=[item for sublist in ThetaCAX for item in sublist]    
    ThetaCAY=[item for sublist in ThetaCAY for item in sublist]
    DMagsCAG=[item for sublist in DMagsCAG for item in sublist]
    DMagsCAX=[item for sublist in DMagsCAX for item in sublist]
    DMagsCAY=[item for sublist in DMagsCAY for item in sublist]
    RadiusCAG=[item for sublist in RadiusCAG for item in sublist]
    RadiusCAX=[item for sublist in RadiusCAX for item in sublist]
    RadiusCAY=[item for sublist in RadiusCAY for item in sublist]
    
    #compute remaining two values of interest
    numUnitsPerPeriodCAG=[ 360.0/theta for theta in ThetaCAG]
    numUnitsPerPeriodCAX=[ 360.0/theta for theta in ThetaCAX]
    numUnitsPerPeriodCAY=[ 360.0/theta for theta in ThetaCAY]
    
    TruePeriodCAG=[ (nUPP+1.0)*D for nUPP,D in zip(numUnitsPerPeriodCAG,DMagsCAG)]
    TruePeriodCAX=[ (nUPP+1.0)*D for nUPP,D in zip(numUnitsPerPeriodCAX,DMagsCAX)]
    TruePeriodCAY=[ (nUPP+1.0)*D for nUPP,D in zip(numUnitsPerPeriodCAY,DMagsCAY)]
    
    return [ [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r] for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r in zip(ResG, ResX, ResY, ThetaCAG, ThetaCAX, ThetaCAY, DMagsCAG, DMagsCAX, DMagsCAY, RadiusCAG, RadiusCAX, RadiusCAY,TruePeriodCAG,TruePeriodCAX,TruePeriodCAY, numUnitsPerPeriodCAG, numUnitsPerPeriodCAX, numUnitsPerPeriodCAY)]
    
def readUnitSymmetry(atoms):
    '''Function takes atomic coords and analyses them to determine the helical params of a structure assuming it is collagen.
        Uses same methodology as readSymmetryFromAtoms.  However the results are returned as an array of values in which
        each entry corresponds to the helical parameters for each GXY repeating unit in the structure.  
        Returns the list of GXY residues indices associated with each set of values to aid identification'''
    
    #always ignore four residues at end of chain
    capRes=4
    
    #set up output arrays
    AllAtomList=[]
    NGList=[]
    CAGList=[]
    CAXList=[]
    CAYList=[]    
    outputResiduesList=[]
    
    #first move the atoms to the COM frame. Do everything there.
    [COM, newAtoms]=atomsToCOM(atoms)
    
    #identify the chains in the atoms
    chains=breakAtomsInToChains(newAtoms)
    
    #generate a list of residues in each chain
    residuesInChains=[ findResidues(chain) for chain in chains]
    
    #generate a GXY letter for each residue
    GXYList =[ idGXYPatternAll(chain) for chain in residuesInChains]
    
    #group the atoms in each residues into sublists
    residueList=[ breakChainIntoResidues(chain) for chain in chains ]
    
    #Extract arrays of useful atoms coords:
    #    the back bone of each chain (N,CA,C)
    #    The Ns of each Gly
    #    The CAs of each Gly, Pro (X) and Hyp (Y)
        
    for residues,GXYInfo in zip(residueList,GXYList):
        
        #create arrays to store the atoms of the coords we are interested in in the current residue
        atomicCoords=[]
        CAG=[]
        CAX=[]
        CAY=[]
        NG=[]
        curResNum=0

        #Ignore the first capRes residues and add an extra GXY repeat at the end when computing the NG array
        #Need the extra entry in the NG array because theta and Z are computed as differences.
        for curRes, GXY in zip(residues[capRes:-(capRes-3):1],GXYInfo[capRes:-(capRes-3):1]):
            #extract arrays of atoms of interest
            for atom in curRes:
                if (atom[1] in ['N']) and (GXY in ['G']):
                    NG.append(np.array([atom[7],atom[8],atom[9]]))                    

        #Ignore capRes residues at either end of the array
        for curRes, GXY in zip(residues[capRes:-capRes:1],GXYInfo[capRes:-capRes:1]):
            
            #remember the Ids of the residues in the current repeating unit in sets of threes.
            if curResNum==0:
                curResList=[curRes[0][5],curRes[0][5]+1,curRes[0][5]+2]
                outputResiduesList.append(curResList)
            curResNum+=1
            if curResNum==3:
                curResNum=0
            
            #extract arrays of atoms of interest
            for atom in curRes:
                if atom[1] in ['N','C','CA']:
                    atomicCoords.append(np.array([atom[7],atom[8],atom[9]]))  
                if atom[1] in ['CA']:
                    if GXY in ['G']:
                        CAG.append(np.array([atom[7],atom[8],atom[9]]))
                    if GXY in ['X']:
                        CAX.append(np.array([atom[7],atom[8],atom[9]]))
                    if GXY in ['Y']:
                        CAY.append(np.array([atom[7],atom[8],atom[9]]))
                        
        #remember each chain of atomic coords separately
        AllAtomList.append(atomicCoords)
        NGList.append(NG)
        CAGList.append(CAG)
        CAXList.append(CAX)
        CAYList.append(CAY)

    #Take the mean of the first atom in each chain to be the base of the helix
    basePoint3Space=np.mean([AllAtomList[0][0],AllAtomList[1][0],AllAtomList[2][0]],0)

    #Take the radius guess as the average of the distances from the basePoint3Space to the start of each chain
    radiusGuess=np.mean([np.linalg.norm(AllAtomList[0][0]-basePoint3Space),np.linalg.norm(AllAtomList[1][0]-basePoint3Space),np.linalg.norm(AllAtomList[2][0]-basePoint3Space)])

    #Take an initial guess at the axis as the vector between basepoint and the centre of mass: COM=([0,0,0])
    zVecGuess=(np.array([0.0,0.0,0.0])-basePoint3Space)/np.linalg.norm(basePoint3Space) 

    #compute point at z=0 for line through basepoint in direction of zVecGuess. The length of the cylinder is irrelevant. Thus we can use the point where the axis crosses the yx plane to define the base point.
    #This eliminates another parameter for this fit. Only a problem if the z-axis is parallel to the xy plane. we'll cross that bridge when it turns up. 
    #basePoint2Space=array([-1.0*ZVecMeanNorm[0]*basePoint3Space[2]/ZVecMeanNorm[2]+basePoint3Space[0],-1.0*ZVecMeanNorm[1]*basePoint3Space[2]/ZVecMeanNorm[2]+basePoint3Space[1],0])
    basePoint2Space=np.array([-1.0*zVecGuess[0]*basePoint3Space[2]/zVecGuess[2]+basePoint3Space[0],-1.0*zVecGuess[1]*basePoint3Space[2]/zVecGuess[2]+basePoint3Space[1],0])
    
    #compute the polar coords of the Z-vector - only need the orientation as this is always a normal vector - reduces params by one.
    #ignore rZHat
    [rZHat,phiZHat,thetaZHat]=XYZTo3DPolar(zVecGuess)
    
    #Now have estimates for ZVector, the basepoint and the radius which defines a cylinder. Set it up as an array of parameters    
    InitialGuess=[radiusGuess,thetaZHat,phiZHat,basePoint2Space[0],basePoint2Space[1]]

    #flatten the atom list so we just have a single list of the all the back bone atoms vectors.
    AllAtomsListFlat=[item for sublist in AllAtomList for item in sublist]

    #find the overall range of the data vectors
    maxVals=np.amax(AllAtomsListFlat,0)
    minVals=np.amin(AllAtomsListFlat,0)
    
    #the radius ought not to be larger than half the longest possible distance within the cloud of data and must be greater than 0. whacked in a 20 because collagen is long and thin and fitting was being silly.
    radiusRange=np.linalg.norm(maxVals-minVals)/10
    
    #The base point parameter on the xy plane should not move more than radiusRange in any direction from the initial basepoint guess 
    
    #Set the bounds; the phi and theta bounds are set to be a more than pi to allow wrap arounds. It tended to get stuck at pi and not know that it could move beyond... seems to work...
    #If the gradient is going up when a parameter hits the boundary it stays there. If it can go past the boundary and the gradient becomes -ve it jumps away from the boundary...
    bounds=[(0,radiusRange),(0,2*np.pi),(-2*np.pi,2*np.pi),(basePoint2Space[0]-radiusRange,basePoint2Space[0]+radiusRange),(basePoint2Space[1]-radiusRange,basePoint2Space[1]+radiusRange)]
    
    #create a function pointer - [] is for plotting out stuff on the fly to monitor progress if desired
    f=(lambda x: radialDistanceProjectionSum(x,AllAtomsListFlat,[]))
    
    #minimise the RadialDistanceProjection sum starting with the initial guess
    finalFit=fmin_l_bfgs_b(f, InitialGuess, approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0)
    #print('initial Z Vector Guess: ', zVecGuess)
    #print('Initial params guess (radius,theta,phi,basePointX,basePointY): ', InitialGuess)
    #print('Final params and errors:', finalFit)
    
    #construct the final vectors for the helix axis.
    zFinal=ThreeDPolarToXYZ(1.0,float(finalFit[0][1]),float(finalFit[0][2]))
    basePointFinal=np.array([finalFit[0][3],finalFit[0][4],0])

    #Now we have the z-axis vector and a defined point on the axis we can compute interesting things

    #generate output arrays
    Theta=[]
    DMags=[]
    RadiusCAG=[]
    RadiusCAX=[]
    RadiusCAY=[]

    #We previously extract four reference points from each GXY repeating unit
    # The N of Gly and the CAlphas for Gly, X and Y positions. 
    # Can use to analyse the rotation and spacing along the structure. 
    # Look at each chain separately.
    for NG in NGList:
        #compute position of each point relative to final basePoint
        basePointCoordsNG= [point-basePointFinal for point in NG]
        #project the data onto the zAxis
        zComponentsNG=[np.dot(point,zFinal) for point in basePointCoordsNG]
        #project the point onto the base plane
        basePlaneVectorNG=[ point - zComp*zFinal for point, zComp in zip(basePointCoordsNG,zComponentsNG)]        
        #Theta is the angle between each adjacent base plane Vectors in the same strand; return in degrees
        Theta.append([np.arccos(np.dot(bPlaneVec/np.linalg.norm(bPlaneVec),basePlaneVectorNG[bvCurIndex+1]/np.linalg.norm(basePlaneVectorNG[bvCurIndex+1])))*180.0/np.pi for bvCurIndex, bPlaneVec in enumerate(basePlaneVectorNG[:-1])])
        #Vertical displacement between units is the difference between each ZComponent
        DMags.append([abs(zComponentsNG[zCurIndex+1]-curZ) for zCurIndex, curZ in enumerate(zComponentsNG[:-1])])

    #compute the RadiusCAG, CAX and CAY for each repeat unit in each chain        
    for CAG,CAX,CAY in zip(CAGList,CAXList,CAYList):
        #compute position of each point relative to final basePoint
        basePointCoordsCAG= [point-basePointFinal for point in CAG]        
        basePointCoordsCAX= [point-basePointFinal for point in CAX]
        basePointCoordsCAY= [point-basePointFinal for point in CAY]
        #project the data onto the zAxis
        zComponentsCAG=[np.dot(point,zFinal) for point in basePointCoordsCAG]
        zComponentsCAX=[np.dot(point,zFinal) for point in basePointCoordsCAX]
        zComponentsCAY=[np.dot(point,zFinal) for point in basePointCoordsCAY]
        #project the point onto the base plane
        basePlaneVectorCAG=[ point - zComp*zFinal for point, zComp in zip(basePointCoordsCAG,zComponentsCAG)]
        basePlaneVectorCAX=[ point - zComp*zFinal for point, zComp in zip(basePointCoordsCAX,zComponentsCAX)]
        basePlaneVectorCAY=[ point - zComp*zFinal for point, zComp in zip(basePointCoordsCAY,zComponentsCAY)]
        #Radius of the different CAlphas given by magnitude of the basePlaneVectors
        RadiusCAG.append([np.linalg.norm(bpvCAG) for bpvCAG in basePlaneVectorCAG])
        RadiusCAX.append([np.linalg.norm(bpvCAX) for bpvCAX in basePlaneVectorCAX])
        RadiusCAY.append([np.linalg.norm(bpvCAY) for bpvCAY in basePlaneVectorCAY])
        
    #collapse the outermost chain structure for all the output arrays
    Theta=[item for sublist in Theta for item in sublist]
    DMags=[item for sublist in DMags for item in sublist]
    RadiusCAG=[item for sublist in RadiusCAG for item in sublist]
    RadiusCAX=[item for sublist in RadiusCAX for item in sublist]
    RadiusCAY=[item for sublist in RadiusCAY for item in sublist]
    
    #compute remaining two values of interest
    numUnitsPerPeriod=[ 360.0/theta for theta in Theta]
    TruePeriod=[ nUPP*D for nUPP,D in zip(numUnitsPerPeriod,DMags)]        
    
    return [outputResiduesList, Theta, DMags, RadiusCAG, RadiusCAX, RadiusCAY,TruePeriod,numUnitsPerPeriod]
    
def readSymmetryFromAtoms(atoms,capRes,plotFig):
    '''Function takes the atomic coords and analyses them to determine the helical parameters of a structure, assuming it is collagen.
    Sort of based on the logic in Sugeta and Miyazawa 1967. And adapted based on experience in analyseH used to compute L and R in JPCB 117,26.
    Returns standard helical parameters for collagen. capRes defines how many residues to ignore at the beginning and end of each chain'''

    #create figures and windows for outputting graphics if required

    #axis fitting
    fig1=[]
    if plotFig==1:
        fig1 = plt.figure()

    #final axis fitting
    fig2=[]
    if plotFig==2:
        fig2 = plt.figure()

    #radius fitting
    fig3=[]
    if plotFig==3:
        fig3 = plt.figure()

    #chain to chain
    fig4=[]
    if plotFig==4:
        fig4 = plt.figure()

    #setup output arrays
    vecListA=[]
    vecListB=[]
    vecListCol=[]
    AllAtomList=[]
    CAlphaGlyList=[]
    Theta=[]
    DMags=[]
    lengthAlongAxis=[]
    
    
    #first move the atoms to the COM frame. Do everything there.
    [COM,newAtoms]=atomsToCOM(atoms)
    
    #identify the chains in the atoms
    chains=breakAtomsInToChains(newAtoms)
    
    #assume the GXY pattern repeats every three residues, for each chain get a list of the N atom coords in every third residue starting at the zeroth
    #extract all the atoms coords in the back bone of each chain and remember as each chain individually - ignore cap Res residues at each end
    residueList=[]
    for chain in chains:
        #identify the residues within each chain
        residueList.append(breakChainIntoResidues(chain))
    
    
    for residues in residueList:
        #create an array to store the atoms of the coords we are interested in
        atomicCoords=[]
        CAlphaGly=[]
        #check for capRes==0 - don't ignore anything:
        if capRes==0:
            #Loop through each residue, ignoring capres residues at either end, and obtain the atomic coords of 
            #the interesting atoms. 
            for curRes in residues[0::1]:
                for atom in curRes:
                    if atom[1] in ['N','C','CA']:
                        atomicCoords.append(np.array([atom[7],atom[8],atom[9]]))
                    if atom[3] in ['GLY']:
                        if atom[1] in ['CA']:
                            CAlphaGly.append(np.array([atom[7],atom[8],atom[9]]))
                          
        else:
            #if capRes is non-zero then ignore capRes residues at either end of the array
            for curRes in residues[capRes:-capRes:1]:
                for atom in curRes:
                    if atom[1] in ['N','C','CA']:
                        atomicCoords.append(np.array([atom[7],atom[8],atom[9]]))  
                    if atom[3] in ['GLY']:
                        if atom[1] in ['CA']:
                            CAlphaGly.append(np.array([atom[7],atom[8],atom[9]]))

        #remember each chain of atomic coords separately
        AllAtomList.append(atomicCoords)
        CAlphaGlyList.append(CAlphaGly)
        #Construct a list of the tangent T vectors between the repeating units of the inner helix. E.g. the Nitrogen atoms in the GLY residues. 
        #important thing is to Pick the same single point in each repeating unit.
        #TVecs=[atomicCoords[i + 1] - x for i, x in enumerate(atomicCoords[:-1])]
        
        #Normalise the Tangent Vecs
        #TVecsNorm=[ tv/linalg.norm(tv) for tv in TVecs]
        
        #compute Bi-Vectors from the Tangent vectors. Start at first point where there are two vectors defined. Current T - next T 
        #BVecs=[TVecsNorm[tvCurIndex + 1]- tvCur for tvCurIndex, tvCur in enumerate(TVecsNorm[0:-1])]
        
        #Normalise the Outer Bi-Vectors
        #BVecsNorm=[ bv/linalg.norm(bv) for bv in BVecs]
        
        #testCPOA
        #a=[array([1,0,0]),array([2,2,2]),array([2,2,4])]
        #b=[array([2,0,0]),array([2,2,3]),array([2,2,10])]
        #c=[array([0,1,1]),array([2,2.5,2]),array([4,2.5,2])]
        #d=[array([0,2,1]),array([2.5,2.5,2]),array([10,2.5,2])]
        #outArray=cpoa(a,b,c,d)
        
        #Estimate the point where each bivector meets the helix axis. This is the closest point of approach of adjacent bivectors. Works with lists of vectors.
        #[rho1List,paList,rho2List,pbList]=cpoa(atomicCoords[1:-2],[a+b for a,b in zip(atomicCoords[1:-2],BVecsNorm[0:-1])],atomicCoords[2:-1],[a+b for a,b in zip(atomicCoords[2:-1],BVecsNorm[1:])])
        
        #create a growing list of estimates of helix radius
        #rhoList.append(rho1List)
        #rhoList.append(rho2List)
        
        #Create unit vectors between the points of closest approach of the bivectors. These are estimates of the helix axis vector.
        #ZVecsHat.append([ (pb-pa)/linalg.norm(pb-pa) for pb,pa in zip(pbList,paList)])

        #grow a vector list to plot vectors of interest for debugging
        vecListA+=atomicCoords[:-1]
        vecListB+=atomicCoords[1:]
        vecListCol+=['k']*len(atomicCoords[1:])
        
        #vecListA+=atomicCoords[:-1]
        #vecListB+=[ a+b for a,b in zip(atomicCoords[:-1],TVecsNorm)]
        #vecListCol+=['r']*len(TVecsNorm)
        
        #vecListA+=atomicCoords[1:-1]
        #vecListB+=[ a+b for a,b in zip(atomicCoords[1:-1],BVecsNorm)]
        #vecListCol+=['b']*len(BVecsNorm)
        
        #vecListA+=atomicCoords[1:-2]
        #vecListB+=paList
        #vecListCol+=['g']*len(paList)

        #vecListA+=atomicCoords[2:-1]
        #vecListB+=pbList
        #vecListCol+=['m']*len(pbList)

        #vecListA+=paList
        #vecListB+=pbList
        #vecListCol+=['c']*len(pbList)

        #vecListA+=paInnerList
        #vecListB+=pbInnerList
        #vecListCol+=['r']*len(pbInnerList)

    #flatten the zVectors List and take the mean
    #ZVecMean=mean([item for sublist in ZVecsHat for item in sublist],0)
        
    #Normalise the mean ZVector
    #ZVecMeanNorm=ZVecMean/linalg.norm(ZVecMean)

    #compute the mean radius of the helix    
    #rhoMean=mean([item for sublist in rhoList for item in sublist])

    #Take the mean of the first atom in each chain to be the base of the helix
    basePoint3Space=np.mean([AllAtomList[0][0],AllAtomList[1][0],AllAtomList[2][0]],0)

    #Take the radius guess as the average of the distances from the basePoint3Space to the start of each chain
    radiusGuess=np.mean([np.linalg.norm(AllAtomList[0][0]-basePoint3Space),np.linalg.norm(AllAtomList[1][0]-basePoint3Space),np.linalg.norm(AllAtomList[2][0]-basePoint3Space)])

    #Take an initial guess at the axis as the vector between basepoint and the centre of mass: COM=([0,0,0])
    zVecGuess=(np.array([0.0,0.0,0.0])-basePoint3Space)/np.linalg.norm(basePoint3Space) 

    #compute point at z=0 for line through basepoint in direction of zVecGuess. The length of the cylinder is irrelevant. Thus we can use the point where the axis crosses the yx plane to define the base point.
    #This eliminates another parameter for this fit. Only a problem if the z-axis is parallel to the xy plane. we'll cross that bridge when it turns up. 
    #basePoint2Space=array([-1.0*ZVecMeanNorm[0]*basePoint3Space[2]/ZVecMeanNorm[2]+basePoint3Space[0],-1.0*ZVecMeanNorm[1]*basePoint3Space[2]/ZVecMeanNorm[2]+basePoint3Space[1],0])
    basePoint2Space=np.array([-1.0*zVecGuess[0]*basePoint3Space[2]/zVecGuess[2]+basePoint3Space[0],-1.0*zVecGuess[1]*basePoint3Space[2]/zVecGuess[2]+basePoint3Space[1],0])
    
    #add the ZVecGuess and initial basePoint 3Space to the plot list   
    vecListA+=[basePoint3Space]
    vecListB+=[basePoint3Space+10*zVecGuess]
    vecListCol+=['c']
    
    #add the ZVecGuess and initial basePlot 2Space to the plot list
    vecListA+=[basePoint2Space]
    vecListB+=[basePoint2Space+10*zVecGuess]
    vecListCol+=['r']

    #compute the polar coords of the Z-vector - only need the orientation as this is always a normal vector - reduces params by one.
    #ignore rZHat
    [rZHat,phiZHat,thetaZHat]=XYZTo3DPolar(zVecGuess)
    
    #Now have estimates for ZVector, the basepoint and the radius which defines a cylinder. Set it up as an array of parameters    
    InitialGuess=[radiusGuess,thetaZHat,phiZHat,basePoint2Space[0],basePoint2Space[1]]

    #speeda things up a little! :) This is a hack from the output of the first fit using the above to speed up debugging...
    #InitialGuess=[2.85011399,  2.82530913, 0.24370711,  0.46294744,  0.49068788]

    #flatten the atom list so we just have a single list of the all the back bone atoms vectors.
    AllAtomsListFlat=[item for sublist in AllAtomList for item in sublist]

    #find the overall range of the data vectors
    maxVals=np.amax(AllAtomsListFlat,0)
    minVals=np.amin(AllAtomsListFlat,0)
    
    #the radius ought not to be larger than half the longest possible distance within the cloud of data and must be greater than 0. whacked in a 20 because collagen is long and thin and fitting was being silly.
    radiusRange=np.linalg.norm(maxVals-minVals)/10
    
    #The base point parameter on the xy plane should not move more than radiusRange in any direction from the initial basepoint guess 
    
    #Set the bounds; the phi and theta bounds are set to be a more than pi to allow wrap arounds. It tended to get stuck at pi and not know that it could move beyond... seems to work...
    #If the gradient is going up when a parameter hits the boundary it stays there. If it can go past the boundary and the gradient becomes -ve it jumps away from the boundary...
    bounds=[(0,radiusRange),(-0.1*np.pi,2.5*np.pi),(-2.5*np.pi,2.5*np.pi),(basePoint2Space[0]-radiusRange,basePoint2Space[0]+radiusRange),(basePoint2Space[1]-radiusRange,basePoint2Space[1]+radiusRange)]
    
    #create a function pointer - fig is for plotting out stuff on the fly to monitor progress if desired
    f=(lambda x: radialDistanceProjectionSum(x,AllAtomsListFlat,fig1))
    
    #minimise the RadialDistanceProjection sum starting with the initial guess
    finalFit=fmin_l_bfgs_b(f, InitialGuess, approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0)
    #print('initial Z Vector Guess: ', zVecGuess)
    #print('Initial params guess (radius,theta,phi,basePointX,basePointY): ', InitialGuess)
    #print('Final params and errors:', finalFit)
    
    #output the final plot
    if plotFig==2:
        fig2=plt.figure()
        radialDistanceProjectionSum(finalFit[0],AllAtomsListFlat,fig2)
    
    #construct the final vectors for the helix axis.
    zFinal=ThreeDPolarToXYZ(1.0,float(finalFit[0][1]),float(finalFit[0][2]))
    basePointFinal=np.array([finalFit[0][3],finalFit[0][4],0])

    #add the final Z and basePoint to the plot list (after translating out of COM frame)   
    vecListA+=[basePointFinal]
    vecListB+=[basePointFinal+10*zFinal]
    vecListCol+=['g']


    #Now we have the z-axis vector and a defined point on the axis we can compute interesting things
    
    #create an array to store the atoms of the coords we are interested in
    CAlphaGly=[]
    CAlphaPro=[]
    CAlphaHyp=[]

    #Find the three radii of carbon alphas - retain the ca glys for the chain fitting
    for residues in residueList:

        #check for capRes==0 - don't ignore anything:
        if capRes==0:
            #Loop through each residue, ignoring capres residues at either end, and obtain the atomic coords of 
            #the interesting atoms. 
            for curRes in residues[0::1]:
                for atom in curRes:
                    if atom[1] in ['CA']: 
                        if atom[3] in ['GLY']:
                            CAlphaGly.append(np.array([atom[7],atom[8],atom[9]]))
                        if atom[3] in ['PRO']:
                            CAlphaPro.append(np.array([atom[7],atom[8],atom[9]]))
                        if atom[3] in ['HYP']:
                            CAlphaHyp.append(np.array([atom[7],atom[8],atom[9]]))
        else:
            #if capRes is non-zero then ignore capRes residues at either end of the array
            for curRes in residues[capRes:-capRes:1]:
                for atom in curRes:
                    if atom[1] in ['CA']: 
                        if atom[3] in ['GLY']:
                            CAlphaGly.append(np.array([atom[7],atom[8],atom[9]]))
                        if atom[3] in ['PRO']:
                            CAlphaPro.append(np.array([atom[7],atom[8],atom[9]]))
                        if atom[3] in ['HYP']:
                            CAlphaHyp.append(np.array([atom[7],atom[8],atom[9]]))

    #Now fit the radius
    bounds=[(0,radiusRange)]
    
    #create a function pointer
    alphaGly=(lambda x: radialDistanceProjectionSumKnownAxis(x,basePointFinal,zFinal,CAlphaGly,fig3))
    alphaHyp=(lambda x: radialDistanceProjectionSumKnownAxis(x,basePointFinal,zFinal,CAlphaHyp,fig3))
    alphaPro=(lambda x: radialDistanceProjectionSumKnownAxis(x,basePointFinal,zFinal,CAlphaPro,fig3))
    
    #minimise the RadialDistanceProjection sum starting with the initial guess
    GlyRadiusData=fmin_l_bfgs_b(alphaGly, [finalFit[0][0]], approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0)
    HypRadiusData=fmin_l_bfgs_b(alphaHyp, [finalFit[0][0]], approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0) 
    ProRadiusData=fmin_l_bfgs_b(alphaPro, [finalFit[0][0]], approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0)
    
    #print('Gly Radius:', GlyRadiusData[0][0])
    #print('Hyp Radius:', HypRadiusData[0][0])
    #print('Pro Radius:', ProRadiusData[0][0])
   
    #Find the chain angle and displacement
    bounds=[(-2.5*np.pi,2.5*np.pi),(-10,10)]
    
    #create a function pointer
    Chain1To2=(lambda x: RMSDBetweenChains(x,basePointFinal,zFinal,CAlphaGlyList[0],CAlphaGlyList[1],fig4))
    Chain2To3=(lambda x: RMSDBetweenChains(x,basePointFinal,zFinal,CAlphaGlyList[1],CAlphaGlyList[2],fig4))
    Chain3To1=(lambda x: RMSDBetweenChains(x,basePointFinal,zFinal,CAlphaGlyList[2], CAlphaGlyList[0],fig4))
    
    #minimise the RadialDistanceProjection sum starting with the initial guess
    Chain1To2Params=fmin_l_bfgs_b(Chain1To2, [-100*np.pi/180,-2.8], approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0)
    Chain2To3Params=fmin_l_bfgs_b(Chain2To3, [-100*np.pi/180,-2.8], approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0) 
    Chain3To1Params=fmin_l_bfgs_b(Chain3To1, [-100*np.pi/180,-2.8], approx_grad=True, bounds=bounds, factr=10, epsilon=1e-10, maxfun=1000, disp=0)
    
    #print("Chain1To2Params: ",Chain1To2Params[0][0]*180.0/pi,Chain1To2Params )
    #print("Chain2To3Params: ",Chain2To3Params[0][0]*180.0/pi,Chain2To3Params)
    #print("Chain3To1Params: ",Chain3To1Params[0][0]*180.0/pi,Chain3To1Params)
    
    DeltaZChain=(Chain2To3Params[0][1]+Chain3To1Params[0][1])/2.0
    DeltaThetaChain=(180/np.pi)*(Chain2To3Params[0][0]+Chain3To1Params[0][0])/2.0
    
    #print("DeltaZChain; ",DeltaZChain)
    #print("DeltaThetaChain: ", DeltaThetaChain)

    #extract the same reference point from each GXY repeating unit and use to analyse the rotation and spacing along the structure. 
    #Look at each chain separately. Typically use N of the Gly as reference point.
    for residues in residueList:

        #create an array to store the atoms of the coords we are interested in
        atomicCoords=[]
        
        #check for capRes==0 - don't ignore anything:
        if capRes==0:
            #Loop through each residue, ignoring capres residues at either end, and obtain the atomic coords of 
            #the interesting atoms. 
            for curRes in residues[0::3]:
                for atom in curRes:
                    if atom[1] in ['N']:
                        atomicCoords.append(np.array([atom[7],atom[8],atom[9]]))  
        else:
            #if capRes is non-zero then ignore capRes residues at either end of the array
            for curRes in residues[capRes:-capRes:3]:
                for atom in curRes:
                    if atom[1] in ['N']:
                        atomicCoords.append(np.array([atom[7],atom[8],atom[9]]))  


        #compute position of each point relative to final basePoint
        basePointCoords= [point-basePointFinal for point in atomicCoords]
        
        vecListA+=[basePointFinal]*len(basePointCoords)
        vecListB+=[basePointFinal+bpCoord for bpCoord in basePointCoords]
        vecListCol+=['m']*len(basePointCoords)
        
        #project the data onto the zAxis
        zComponents=[np.dot(point,zFinal) for point in basePointCoords]
        
        vecListA+=[basePointFinal]*len(zComponents)
        vecListB+=[basePointFinal+z*zFinal for z in zComponents]
        vecListCol+=['c']*len(zComponents)
        
        #project the point onto the base plane
        basePlaneVector=[ point - zComp*zFinal for point, zComp in zip(basePointCoords,zComponents)]
        
        vecListA+=[basePointFinal + bpCoord for bpCoord in basePointCoords]
        vecListB+=[basePointFinal + bpCoord - bpVec for bpCoord,bpVec in zip(basePointCoords,basePlaneVector)]
        vecListCol+=['g']*len(basePlaneVector)
        
        #Theta is the angle between each adjacent base plane Vectors in the same strand; return in degrees
        Theta.append([np.arccos(np.dot(bPlaneVec/np.linalg.norm(bPlaneVec),basePlaneVector[bvCurIndex+1]/np.linalg.norm(basePlaneVector[bvCurIndex+1])))*180.0/np.pi for bvCurIndex, bPlaneVec in enumerate(basePlaneVector[:-1])])
    
        #Vertical displacement between units is the difference between each ZComponent
        DMags.append([abs(zComponents[zCurIndex+1]-curZ) for zCurIndex, curZ in enumerate(zComponents[:-1])])
        
        #compute total length along axis from first to last unit
        lengthAlongAxis.append(abs(max(zComponents)-min(zComponents)))
        
    #compute some useful final output values
    #print("Individual theta values for each strand:")
    #print(Theta[0])
    #print(Theta[1])
    #print(Theta[2])
    #print("mean theta values per strand")
    #print(mean(Theta[0]))
    #print(mean(Theta[1]))
    #print(mean(Theta[2]))
    meanTheta=np.mean([item for sublist in Theta for item in sublist])
    meanD=np.mean([item for sublist in DMags for item in sublist])
    meanLengthAlongAxis=np.mean(lengthAlongAxis)
    numUnitsPerPeriod=360.0/meanTheta
    TruePeriod=numUnitsPerPeriod*meanD        
    #print('Final Z Axis: ', zFinal)
    #print('meanTheta: ',meanTheta)
    #print('meanD:',meanD)
    #print('LengthAlongAxis: ', meanLengthAlongAxis)
    #print('TruePeriod: ', TruePeriod)
    #print('numUnitsPerPeriod:', numUnitsPerPeriod)
    
    #plot all the vectors of interest in a new window 
    if plotFig > 0:        
        fig2 = plt.figure()
        if plotFig==2:
            plotVectors(fig2,vecListA,vecListB,vecListCol,False)
        if plotFig==3:
            plotVectors(fig2,vecListA,vecListB,vecListCol,True)

    return [meanTheta,meanD,GlyRadiusData[0][0],HypRadiusData[0][0],ProRadiusData[0][0],meanLengthAlongAxis,TruePeriod,numUnitsPerPeriod,DeltaZChain,DeltaThetaChain]


def ThreeDPolarToXYZ(r,theta,phi):
    x=r*np.sin(phi)*np.cos(theta)
    y=r*np.sin(phi)*np.sin(theta)
    z=r*np.cos(phi)
    return np.array([x,y,z])
    
def XYZTo3DPolar(v):
    r=np.linalg.norm(v)
    vHat=v/r
    phi=np.arccos(vHat[2])
    theta=np.arctan2(vHat[1],vHat[0])
    return np.array([r,theta,phi])

def RMSDBetweenChains(x,basePoint,ZVec,list1,list2,fig):
    
    #move chains to helix space
    list1BasePoint=[l-basePoint for l in list1]
    list2BasePoint=[l-basePoint for l in list2]
    
    #normalise Z vector
    ZVecNorm=ZVec/np.linalg.norm(ZVec)
    
    #extract parameters
    deltaTheta, deltaZ=x
    
    #Find Z components of list2BasePoint
    zComp=[np.dot(dataPoint,ZVecNorm) for dataPoint in list2BasePoint]
    
    #project each data Vector onto a plane perpendicular to the ZVector 
    RVecs=[dataPoint-z*ZVecNorm for dataPoint,z in zip(list2BasePoint,zComp)]
    
    #Compute two basis vectors in the plane
    basisVec1=RVecs[0]/np.linalg.norm(RVecs[0])
    basisVec2=np.cross(basisVec1,ZVecNorm)
    basisVec2=basisVec2/np.linalg.norm(basisVec2)

    #Rotate basis vectors
    basisVec1New= np.cos(deltaTheta)*basisVec1+np.sin(deltaTheta)*basisVec2
    basisVec2New= -np.sin(deltaTheta)*basisVec1+np.cos(deltaTheta)*basisVec2
    
    #Compute rotated vector for RVecs
    NewRVecs=[ np.dot(r,basisVec1)*basisVec1New + np.dot(r,basisVec2)*basisVec2New for r in RVecs]
    
    #Compute the new overall vector
    list2Shifted=[  (z - deltaZ)*ZVecNorm + r for r,z in zip(NewRVecs,zComp)]
    
    #compare the length of the lists. take the shorter one. Should be the same really.
    numAtoms=min(len(list1),len(list2))
    
    #Compute the sum of the difference between the atoms in the lists
    ssd=sum([np.linalg.norm(l1-l2) for l1, l2 in zip(list1BasePoint[0:numAtoms],list2Shifted[0:numAtoms])])
    
    if fig:
        #generate the Vectors for 'live' plotting
        vectorListA=[]
        vectorListB=[]
        vectorListC=[]
        vectorListA+=list1BasePoint[0:-1]
        vectorListB+=list1BasePoint[1:]
        vectorListC+=['r']*(len(list1BasePoint)-1)
        vectorListA+=list2BasePoint[0:-1]
        vectorListB+=list2BasePoint[1:]
        vectorListC+=['g']*(len(list2BasePoint)-1)
        vectorListA+=list2Shifted[0:-1]
        vectorListB+=list2Shifted[1:]
        vectorListC+=['b']*(len(list2Shifted)-1)
        vectorListA+=[basePoint-basePoint]
        vectorListB+=[basePoint-basePoint+10*ZVec]
        vectorListC+=['c']
        plotVectors(fig,vectorListA,vectorListB,vectorListC,False)
        print(ssd, deltaTheta, deltaZ    )
    
    return ssd 

    

def radialDistanceProjectionSumKnownAxis(radius,basePoint,ZVec,dataPoints,fig):
    #Set the datapoints relative to the basepoint
    dataPointsHelixFrame=[dataPoint-basePoint for dataPoint in dataPoints]
    
    #project each data Vector onto a plane perpendicular to the ZVector 
    RVecs=[dataPoint-np.dot(dataPoint,ZVec)*ZVec for dataPoint in dataPointsHelixFrame]
    
    #Compute the sum of the squared normal distance between each data point and the suggested radius
    ssd=sum([(np.linalg.norm(r)-radius)**2.0 for r in RVecs])
    
    #if fig is defined then compute and plot the output vectors
    if fig:
        #generate the Vectors for 'live' plotting
        vectorListA=[]
        vectorListB=[]
        vectorListC=[]
        vectorListA+=dataPoints[0:-1]
        vectorListB+=dataPoints[1:]
        vectorListC+=['k']*(len(dataPoints)-1)
        vectorListA+=[basePoint]
        vectorListB+=[basePoint+10*ZVec]
        vectorListC+=['g']
        vectorListA+=[basePoint]*len(RVecs)
        vectorListB+=[basePoint+r for r in RVecs]
        vectorListC+=['c']*len(RVecs)
        tAll=np.linspace(-np.pi,np.pi,25)
        xList=[radius*np.cos(t) for t in tAll]
        yList=[radius*np.sin(t) for t in tAll]
        pVect1=RVecs[0]/np.linalg.norm(RVecs[0])
        pVect2=np.cross(ZVec,pVect1)
        pVect2=pVect2/np.linalg.norm(pVect2)
        cpList=[basePoint+ x*pVect1+y*pVect2 for x,y in zip(xList,yList)]
        vectorListA+=cpList
        vectorListB+=cpList[1:]
        vectorListB+=[cpList[0]]
        vectorListC+=['m']*len(cpList)
        plotVectors(fig,vectorListA,vectorListB,vectorListC,False)
        print(ssd,radius)
    return ssd




#Function computes distance of each data point projected normally onto the surface of a given cylinder.
def radialDistanceProjectionSum(params,dataPoints,fig):
    radius,theta,phi,bx,by=params
    
    #construct vector arrays from params - can always find a base point on the xy plane
    basePoint=np.array([bx,by,0])
    ZVec=ThreeDPolarToXYZ(1.0,float(theta),float(phi))
    
    #project each data Vector onto a plane perpendicular to the ZVector 
    RVecs=[dataPoint-np.dot(dataPoint,ZVec)*ZVec for dataPoint in dataPoints]
    
    #project the base vector onto the same plane.
    baseVectorOnPlane=basePoint-np.dot(basePoint,ZVec)*ZVec

    #Compute the sum of the squared normal distance between each data point and the defined cylinder
    ssd=sum([(np.linalg.norm(r-baseVectorOnPlane)-radius)**2.0 for r in RVecs])

    #if fig is defined then compute and plot the output vectors
    if fig:
        #generate the Vectors for 'live' plotting
        vectorListA=[]
        vectorListB=[]
        vectorListC=[]
        vectorListA+=dataPoints[0:-1]
        vectorListB+=dataPoints[1:]
        vectorListC+=['k']*(len(dataPoints)-1)
        vectorListA+=[basePoint]
        vectorListB+=[basePoint+10*ZVec]
        vectorListC+=['g']
        vectorListA+=[baseVectorOnPlane]*len(RVecs)
        vectorListB+=[baseVectorOnPlane+r for r in RVecs]
        vectorListC+=['c']*len(RVecs)
        tAll=np.linspace(-np.pi,np.pi,25)
        xList=[radius*np.cos(t) for t in tAll]
        yList=[radius*np.sin(t) for t in tAll]
        pVect1=RVecs[0]/np.linalg.norm(RVecs[0])
        pVect2=np.cross(ZVec,pVect1)
        pVect2=pVect2/np.linalg.norm(pVect2)
        cpList=[baseVectorOnPlane+ x*pVect1+y*pVect2 for x,y in zip(xList,yList)]
        vectorListA+=cpList
        vectorListB+=cpList[1:]
        vectorListB+=[cpList[0]]
        vectorListC+=['m']*len(cpList)
        plotVectors(fig,vectorListA,vectorListB,vectorListC,False)
        print(ssd,params)
    return ssd

def readSymmetry(infile,capRes,figFlag,outfile):
    #perform required operations
    atoms=readAtoms(infile)

    #initialise output file
    fO=open(outfile,'w')

    data=readSymmetryFromAtoms(atoms,capRes,figFlag)
    
    for item in data:
        fO.write(str(item)+'\n')
        print(item)
        
    fO.close()

    return

def readResidueSymmetryWrapper(infile,outfile):
    '''reads a pdb file and performs collagen symmetry reading operation on it.'''
    #perform required operations
    atoms=readAtoms(infile)

    dataArray=readResidueSymmetry(atoms)
    
    outputArray=[]
    for line in dataArray:
        l=''
        for item in line:
            l+=str(item)+' '
        l+='\n'    
        outputArray.append(l)
            
    #write the data to file
    writeTextFile(outputArray,outfile)
    print("Done!")
    return

def fixchirality(infile, outfile):
    ''' Function reads the chirality of each hydroxyproline and proline in the input file, 
    It corrects those that need correcting inplace in the input list and outputs them in the outfile. 
    If changes are made, the new PDB will undoubtedly need to be relaxed, but this is not performed here.
    Also prints out this list of chiralities and reports which ones were changed.'''
    # perform required operations
    atoms = readAtoms(infile)
     
    # check the chirality of this residue and fix it
    out_list, list_of_changed_residues = checkchirality(atoms, fix_chirality=True)
    
    # output the atoms to the text file specified
    writeAtomsToTextFile(atoms, outfile)
    
    # use the other stuff to get rid of orange lines in Eclipse. Might as well.
    print(out_list)
    if list_of_changed_residues:
        print("bad residues altered:", list_of_changed_residues)
    else:
        print("No Residues Changed")
    
def readchirality(infile, outfile):
    ''' Writes a file which contains a list of the chirality state
    of each hydroxyproline and proline in the input file.
    Also returns a flag indicating if there is a residue which is
    not 2S-4R anywhere in the file. returns a list of which
    residues are not 2S-4R (or just 2S for prolines).'''
    
    # perform required operations
    atoms = readAtoms(infile)
     
    # check the chiralities
    outlist, badChiralRes = checkchirality(atoms)

    # initialise output list
    fO = open(outfile,'w')

    # output the list into a file
    for l in outlist:
        fO.write(l)

    # close the output file 
    fO.close()

    # set the return flag - maintains backwards compatibility
    chiralityGood = 1
    if badChiralRes:
        chiralityGood = 0
        print("Bad Chiral Residues: ", badChiralRes)
    else:
        print("All residues good.")

    return [chiralityGood, badChiralRes]

def checkchirality(atoms, fix_chirality=False):
    ''' Returns a list of the chirality state of each hydroxyproline and proline residue
    in the input atom list. Provides a summary of the residues which are not 2S-4R. 
    If the fix_chirality flag is set then the residues which are incorrect are corrected 
    in place in the input atoms.'''
    
    # set up output lists
    outlist = []
    badChiralRes=[]

    # creates a list of all residue numbers and their type in the atoms list
    residues = findResidues(atoms)

    # go through each residue in list (in format returned by 
    for (residueNum,residueName) in residues:
 
        # generate a list of atoms in the current residue
        curResidue=[atom for atom in atoms if atom[5]==residueNum]

        #if residue is pro check the chirality state
        if residueName in ['PRO']:
            for atom in curResidue:
                if (atom[1]=='N'):
                    NPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='CA'):
                    CAPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='CB'):
                    CBPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='CG'):
                    CGPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='CD'):
                    CDPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='C'):
                    CPOS=np.array([atom[7],atom[8],atom[9]])

            #compute chirality of position 2.
            NCA = NPOS-CAPOS
            CACB = CBPOS-CAPOS
            CAC = CPOS-CAPOS

            N=np.cross(NCA,CACB)
            N=N/np.linalg.norm(N)
            CZ=np.vdot(CAC,N)
            
            # if it's not S then report it as R
            chirality2='S'
            if CZ>=0:
                chirality2='R'

            # output the result to the outlist    
            outlist.append(str(residueNum)+' 2'+chirality2+'\n')

            # if the chirality is 'bad' then make a note of it.
            # If we are fixing the chirality then do so.
            if chirality2=="R":
                badChiralRes.append(residueNum)
                if fix_chirality:
                    N_C = CPOS - NPOS
                    N_CA = CAPOS - NPOS
                    normal = np.cross(N_C, N_CA)
                    normal = normal /np.linalg.norm(normal)
                    invert_atoms_in_plane(curResidue, NPOS, normal)
                    
            ## debug hack to make sure it actually works!
            #if residueNum==11:
            #    N_C = CPOS - NPOS
            #    N_CA = CAPOS - NPOS
            #    normal = cross(N_C, N_CA)
            #    normal = normal / linalg.norm(normal)
            #    invert_atoms_in_plane(curResidue, NPOS, normal)

        if residueName in ['HYP']:
            for atom in curResidue:
                if (atom[1]=='N'):
                    NPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='CA'):
                    CAPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='CB'):
                    CBPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='CG'):
                    CGPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='CD'):
                    CDPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='C'):
                    CPOS=np.array([atom[7],atom[8],atom[9]])
                if (atom[1]=='OD1'):
                    OPOS=np.array([atom[7],atom[8],atom[9]])

            #compute chirality of position 2.
            chirality2='S'
            NCA = NPOS-CAPOS
            CACB = CBPOS-CAPOS
            CAC = CPOS-CAPOS

            N=np.cross(NCA,CACB)
            N=N/np.linalg.norm(N)
            CZ=np.vdot(CAC,N)
            if CZ>=0:
                chirality2='R'

            #compute chirality of position 4.
            chirality4='R'
            CDCG = CGPOS-CDPOS
            CBCG = CGPOS-CBPOS
            OCG = CGPOS-OPOS

            N=np.cross(CDCG,CBCG)
            N=N/np.linalg.norm(N)
            CZ=np.vdot(OCG,N)
            if CZ<0:
                chirality4='S'

            # output the result to the outlist
            outlist.append(str(residueNum) + ' 2' + chirality2 + ' 4' + chirality4 + '\n')


            # debug hack to make sure it actually works!
            #if residueNum==53:
            #    N_C = CPOS - NPOS
            #    N_CA = CAPOS - NPOS
            #    normal = cross(N_C, N_CA)
            #    normal = normal / linalg.norm(normal)
            #    invert_atoms_in_plane(curResidue, NPOS, normal)

            # check what to do based on results of chirality analysis
            if chirality2=="R" or chirality4=="S": 
                # if either of the chiralities is bad then note the residue.
                badChiralRes.append(residueNum)
                
                # if we are fixing the chirality then do so. 
                if fix_chirality:
                    if chirality2=="R" and chirality4=="S": # want 2S-4R so inversion in plane of N-C-CA will correct both.
                        # Invert the residue in the plane formed by the N, C and CA planes.
                        N_C = CPOS - NPOS
                        N_CA = CAPOS - NPOS
                        normal = np.cross(N_C, N_CA)
                        normal = normal / np.linalg.norm(normal)
                        invert_atoms_in_plane(curResidue, NPOS, normal)


                    if chirality2=="S" and chirality4=="S": # want 2S-4R so only invert about the CG
                        # selects the residues bonded to the CG
                        atoms_of_note = []
                        for atom in curResidue:
                            if atom[1] == 'HG2':
                                atoms_of_note.append(atom)
                            if atom[1] == 'OD1':
                                atoms_of_note.append(atom)
                            if atom[1] == 'HO1':
                                atoms_of_note.append(atom)
                        # invert the residues bonded to the CG in the plane formed by the CG, CB and CD atoms.
                        # Use CG as the origin.
                        CG_CD = CDPOS - CGPOS
                        CG_CB = CBPOS - CGPOS
                        normal = np.cross(CG_CD, CG_CB)
                        normal = normal / np.linalg.norm(normal)
                        invert_atoms_in_plane(atoms_of_note, CGPOS, normal)


                    if chirality2=="R" and chirality4=="R": # want 2S-4R so invert about the CG, and then invert the whole residue.
                        # selects the residues bonded to the CG
                        atoms_of_note = []
                        for atom in curResidue:
                            if atom[1] == 'HG2':
                                atoms_of_note.append(atom)
                            if atom[1] == 'OD1':
                                atoms_of_note.append(atom)
                            if atom[1] == 'HO1':
                                atoms_of_note.append(atom)
                        # invert the residues bonded to the CG in the plane formed by the CG, CB and CD atoms.
                        # Use CG as the origin.
                        CG_CD = CDPOS - CGPOS
                        CG_CB = CBPOS - CGPOS
                        normal = np.cross(CG_CD, CG_CB)
                        normal = normal / np.linalg.norm(normal)
                        invert_atoms_in_plane(atoms_of_note, CGPOS, normal)
                        
                        # Invert the residue in the plane formed by the N, C and CA planes.
                        N_C = CPOS - NPOS
                        normal = np.cross(N_C, N_CA)
                        normal = normal / np.linalg.norm(normal)
                        invert_atoms_in_plane(curResidue, NPOS, normal)


        
    # output the results
    return outlist, badChiralRes 

def invert_atoms_in_plane( atom_list, point, normal):
    ''' Function takes list of atoms and reflects their positions in a plane
    which contains the given point and a specified normal vector'''

    # ensure the normal is a unit normal
    n_hat = normal / np.linalg.norm(normal)
        
    # reflect every atom in the list of atoms
    for atom in atom_list:
        # translate atom to be relative to the given point in the plane.
        V = np.array([atom[7], atom[8], atom[9]]) - point
        
        # perform the reflection in the plane.
        V_refl = V - 2 * np.dot(V, n_hat) * n_hat
        
        # update the atom list and add back in the original point.
        atom[7] = V_refl[0] + point[0]
        atom[8] = V_refl[1] + point[1]
        atom[9] = V_refl[2] + point[2]
        
        
        
        
                             
     
    
    
     
