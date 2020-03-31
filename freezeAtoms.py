#!/usr/bin/env python
import pdbLib as pdb

# Main Routine
if __name__ == '__main__':
    """ Collagen 
"""    
    # load the collagenAtoms, and PPG12 data
    atoms = pdb.readAllAtoms('3HR2_leap.pdb')
    linesAll=[str(atom[0])+' ' for atom in atoms if atom[1]=='CA']
    
    i=0
    lines=[]
    lastLineAppend=1
    line='FREEZE '
    while i<len(linesAll):
        lastLineAppend=1
        line+=linesAll[i]
        if i%20==0 and i>0:
            line+='\n'
            lines.append(line)
            lastLineAppend=0
            line='FREEZE '
        i+=1
        
    if lastLineAppend==1:
        lines.append(line)    
    pdb.writeTextFile(lines, '3HR2.freeze')  