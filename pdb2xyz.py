#!/usr/bin/env python
import sys,os
import pdbLib as pdb

#function write atoms as xyz file format
def writeAtomsAsXYZ(atoms,outfilename,comment):
   
   try:
      fH=open(outfilename,'w')
   except:
      print outfilename
      raise Exception, "Unable to open file for output"

   numAtoms=len(atoms)
   fH.write(str(numAtoms)+'\n')
   fH.write(comment+'\n')
   #count number of atoms
   for atom in atoms:
     print atom
     l=atom[1]+" "+str(atom[7])+" "+str(atom[8])+" "+str(atom[9])+'\n'
     fH.write(l)

   fH.close()
   return


#Main Routine
if __name__ == '__main__':

    if len(sys.argv)!=3:
       print "Usage:  pdb2xyz.py infile.pdb outfile.pdb"
       sys.exit(0)

    inFilename=sys.argv[1]
    outFilename=sys.argv[2]

    #load the pdbFile
    atoms=pdb.readAtoms(inFilename)

    #write the xyz file
    writeAtomsAsXYZ(atoms, outFilename,inFilename)
