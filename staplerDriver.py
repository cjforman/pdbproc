'''
Created on Jan 30, 2024

@author: Chris
'''
import sys
import os
import glob
import itertools as iter
import pdbStapler

# writes a staple file for file1 and file2
def writeStapler(file1, file2, stapleFile):
    fileroot1 = os.path.split(file1)[1][0:-4]
    fileroot2 = os.path.split(file2)[1][0:-4]
    
    
    mode = 'pre'
    
    with open(stapleFile, 'r') as infile, open( fileroot1 + '_' + fileroot2 + '.stp', 'w') as outfile:
        for line in infile:
    
            if line[0:10]=="OUTPUTFILE":
                outfile.write("OUTPUTFILE " + fileroot1 + "_" + fileroot2 + '.pdb\n')
            
            elif line[0:8]=="FRAGMENT" and mode=='pre':
                outfile.write("FRAGMENT " + file1 + '\n')
                mode='fragment1'
                
            elif line[0:6]=='STAPLE':
                outfile.write(line)
                mode='staple'

            elif line[0:8]=="FRAGMENT" and mode=='staple':
                outfile.write("FRAGMENT " + file2 +'\n')
                mode='fragment2'
                
            elif line[0:5]=="CHAIN" and mode=='fragment1' and 'output' in fileroot1:
                outfile.write("CHAIN _ 1 3132\n")

            elif line[0:5]=="CHAIN" and mode=='fragment1':
                outfile.write("CHAIN A 1 3132\n")

            elif line[0:5]=="CHAIN" and mode=='staple':
                outfile.write("CHAIN A 1 253\n")

            elif line[0:5]=="CHAIN" and mode=="fragment2" and 'output' in fileroot2:
                outfile.write("CHAIN _ 1 3132\n")

            elif line[0:5]=="CHAIN" and mode=='fragment2' and not 'output' in fileroot2:
                outfile.write("CHAIN A 1 3132\n")

            else:
                outfile.write(line)
            

if __name__ == '__main__':
    
    pdbGlob = glob.glob(os.path.join(sys.argv[1], "*.pdb"))
    
    stapleFile = sys.argv[2]
    
    for file1, file2 in iter.combinations_with_replacement(pdbGlob, 2):
            print(file1, file2)
            writeStapler(file1, file2, stapleFile)
            
            
    with open('staplecommands.ps1','w') as cmdfile:
        stapleGlob = glob.glob("*.stp")
        for stapleFile in stapleGlob:
            cmdfile.write("python -m pdbStapler " + stapleFile + "\n")