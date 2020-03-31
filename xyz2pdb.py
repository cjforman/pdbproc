#!/usr/bin/env python
import sys,os
sys.path.append('/usr/lib64/python2.7/site-packages')
import pdbLib as pdb
import copy as cp

#Main Routine
if __name__ == '__main__':

    if len(sys.argv)!=4:
       print "Usage:  pdb2xyz.py infile.pdb infile.xyz outfile.pdb"
       sys.exit(0)

    inFilenamePdb=sys.argv[1]
    inFilenameXyz=sys.argv[2]
    outFilename=sys.argv[3]

    #load the pdbFile
    atoms=pdb.readAtoms(inFilenamePdb)
    xyz=pdb.readTextFile(inFilenameXyz)

    if len(atoms)!=len(xyz):
        print 'Inconsistent number of atoms'
        sys.exit(0)

    curAtom=0
    for atom in atoms:
#      print atom
      curXyz=xyz[curAtom].split()
#      print curXyz
      atom[7]=float(curXyz[0])
      atom[8]=float(curXyz[1])
      atom[9]=float(curXyz[2])
#      print atom
#      herFlick
      curAtom+=1

    #write the xyz file
    pdb.writeAtomsToTextFile(atoms, outFilename)
