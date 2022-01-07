#!/usr/bin/env python
import sys
import os
import numpy as np
import scipy as sp
import pdbLib as pl

def commandLineProc():

    usage="""
pdb file processor
==================

General Syntax:	pdbproc.py command inpfile1 [inpfile2] [param1] [param2]

where command and inpfile are mandatory. Command is the action to perform and inpfile
is the name of a PDB file in standard format. Each command takes variable number of parameters.

command is one of:
    
    fragmentPDB, fragmentpdb, fpdb, fPDB, or FPDB inpfile resfile 
    
        Generates a series of minipdbs based on the pairs of start and stop residues in resfile.
        if the start residues does not exist then the output starts at the first residue.
        If the end residue does not exist the tail of the PDB is provided from the start residue onwards.
        iF the end residue is before the start residue then no output is returned. 
        
        example resfile
        1 10
        1 100
        35 39

    eliminateCIS eCIS inpfile
	Calls the createCT algorithm to check if the lowest energy pdb has no CIS states
	if it does then it generates a new atomGroups file to spin only the CIS peptide bonds
        and calls CUDAGMIN for a bunch of steps with CISTRANS checking turned off. 
	Loops until the lowest energy minimum has no CIS peptide bonds
       
    centrePDB CPDB or cPDB inpfile
        translates the PDB to it's centre of mass frame and outputs identical pdb with only coords modified.

    generateCTAtomGroups, GCTAG infile
        takes a initial_cis_trans_state file and creates an atomgroups file for all the CIS peptide bonds. Uses information
        in the pdb file pointed at by infile to help construct the atomgroups file
        
    createCTFile, cCTF inpfile thresh
        generates a CIS_TRANS_STATE file for a given PDB
        
    addTermini inpfile N=-30,10 C
        aliases: addTermini, addtermini, AddTermini, at, aT, AT 
        Adds N or C terminus line to PDB (NME, ACE) if N and/or C are present in the params. 
        A dihedral and bond angle pair may be specified for each terminus. 
        otherwise default is 0, 109.
        Bond length is difference between C and CA or CA and N in adjacent residue. 

    'convertseq', 'ConvertSeq', 'cs', 'cS', 'CS', 'Cs']
    cs inpfile seqfile
        Takes an input file which consists of single letter codes of a protein sequence, like a mode 3 sequence file 
        from readSequence. Outputs a file with filename seqfile which has the three letter codes as per mode 2 of readSequence.
        The three letter codes output file is compatible with the modify sequence command.  

    readpucker or rp inpfile [outfile]
        Checks the pucker state of each residue in inpfile.
        looks for inpfile (automatically appends .pdb if it is not there) and
        analyses the pucker state of each residue before outputting data into
        outfile in a format of resId, resName, Endo|Exo. 
        If outfile is omitted readpucker outputs data in inpfile.pucker.

    groupAtomsXYZ, gAX or gax inpfile
        reads the PDB and dumps the atoms XYZ file, but relabels the atom types according 
        to basic atom type. All Cs are called C, all H1, HE1 etc become H.  Reduces number of objects
	loaded into blender

    makeXYZforBlender ir mXB inpfile backboneOnly mode
        reads the PDB and creates and xyz for reading into blender. If backboneOnly is 1, then it dumps 
        the CA residue as an XYZ file. Atoms in each residues labelled according to the mode. 
        1 = functional nature of the amino acid that the CA belongs to which generates a different colour for each type 
        of residue 
        2 = Each of the twenty residues and IMB for PLPs -> A unique atom. 
    
    ramachandran or rmc inpfile
        Makes a ramachandran plot of the pdb file.
        
    readChirality or rc inpfile [outfile]
        Checks the chirality state of each proline and hydroxyproline  
        in the inpfile. Output is resNum with an 2-X or 4-X where x is R or S.
        Generally should be an 2-S 4-R state.  if it is proline then just get a 2-R state.
        If not specified the outfile is a .chi file.

    fixChirality or fc inpfile [outfile]
        Checks the chirality state of each proline and hydroxyproline  
        in the inpfile. Output is resNum with an 2-X or 4-X where x is R or S.
        If the chirality is not a 2-S 4-R state then it is converted to be so
        and rewritten as a PDB. If not specified the output filename is ..._chi.pdb
                    
    writepucker or wp inpfile puckerStateFile [outfile]
        sets the pucker state of each residue in the inpfile pdb to the status specified in puckerStateFile.
        The pucker State file contains lines with the format resiD, resName, Endo|Exo.
        Looks for inpfile (automatically appends .pdb if it is not there).
        A pucker inversion operation is applied to prolines or hydroxyprolines which do not match the
        required state. Outfile is an optional outputfilename. Outfile is omitted the output file name 
        is inpFileRoot.pucker.pdb.
    
    checkPuckerPattern or cpp inpfile 
        prints true or false if the pdb file contains a ring pucker system
        whose endo and exo states match the endo and exo states of the standard collagen fibre.
    
    
    removeAtoms or ra inpfile rulefile [outfile]
        loads a pdb file and removes lines which match the specifications and then rewrites the pdb file. 
        looks first for infile or infile.pdb and loads it. 
        Then looks for the rulefile or rulefile.lines. which has the format:
    
    flipCT fct inpfile cis_trans forcefield
        Takes a cis trans file from gmin, and performs a series of rotations to convert all the cis peptides to trans.
        Then runs it through the PAG process with a noreduce optionto generate a restart file
    
    string integer
        where string is the string which is contained in the ATOM line that qualifies it for removal, 
        eg H, C, GLY or whatever.  integer is the zero based column number in which the text is to appear.
        envisaged for use solely with the ATOM keyword. 
        The rest of the PDB file is output verbatim, except for the lines which qualify.
        if the outfile is not specified then the default is infile_short.pdb.
    
    prepAmberGmin or PAG inpfile rulefile forcefield [prepFile] [paramsFile]
        rulefile can be: 
            "reduce" which uses the reduce program provided with leap to Trim the structure.
            "noreduce" does not edit the structure at all. 
            <filename> which is a is list of rules concerning which objects (residues ot atoms) to omit. 
            item1 field1
            item2 field2
            ...
              wher item is the value and field is the pdb field.  e.g.
              H 2
              TYR 3
              
              which would remove all lines containing an H in field 2 (atom names) and all lines containing TYR in field 3 (residues)  
            
        PAG then performs the following sequence of commands starting from a pdb file
    
            removeAtoms inpfile rulefile inpfile_clean.pdb
            source leaprc.forcefield
            loadamberprep prepFile
            loadamberparams paramsFile
            mol=loadpdb inpfile_clean.pdb
            saveamberparm mol inpfile_preSym.prmtop inpfile.inpcrd
            savepdb mol inpfile_preSym.pdb
            quit
            renameTermini inpfile_preSym.prmtop inpfile_preSym.pdb
            symmetrize forcefield inpfile_preSym.prmtop inpfile.prmtop
            renameTermini inpfile_preSym.prmtop inpfile_preSym.pdb
    
    readSequence or rsq,RSQ  inpDB mode [width] 
        reads the sequence from the PDB and outputs into file with a .seq on the end.
        Enables easy editing of the PDB sequence. 
            Mode = 1 yields a numbered list of residues. 
            Mode = 2 yields a list of three letter residue codes in a format suitable for modifySequence.
            Mode = 3 yields a string of single chars representing each residue with the standard letters.
            Mode = 4 is three letter codes separated by spaces. 
            Mode = 5 uses an optional width parameter to output 1 letter codes in lines width chars long 
                     as per fasta file format. Defaults to 80 if omitted.
    
    modifySequence or MS  inpPDB newSequence startResidue [outputfile]
        Replaces the sequence names in inpPDB with the sequence in the file newSequence
        starting from the startresidue in the input file. The side chain atoms of 
        the old sequence are ignored and only the backbone atoms are output if the residue is different.
        
        Works by reading through inpPDB and dumping each line to the outputfile verbatim.
        The number of residues copied across so far is counted and if the number of residues 
        copied exceeds the value given in startresidue then the procedure begings to rename the residues.
        If the new residue name is different from the existing residue name then only 
        only backbone atoms are output. 
        
        IF the new sequence is longer than the original sequence then the procedure will simply
        ignore the excess new residues. i.e. the final file will always be the same length as inpPDB.
        IF new residues is shorter than inppdb then the last part of the inpPDB will be untouched. although
        the atom numbers will be renumbered.
    
    
    renameTermini or rT inpfile flag [topologyfile]
        looks for the pdb inpfile or inpfile.pdb and loads it. Then it scans for inpfile.prmtop
        or topologyFile and then adds or removes Ns and Cs to the N and C termini of each chain
        in the topologyFile. The topologyFile is overwritten. If Flag is 0 the topology file is
        output without the Ns and Cs. If Flag is 1 the topology file is output with the Ns and Cs.
    
    renumberRes or RN inpfile [startNum] [outfile]
        starts numbering residues at the beginning of a file and everytime it finds a Nitrogen increments
        the residue number. outputs using standard rules.  looksfor inpFile or inpFile.pdb and if outfile
        is not specified writes to inpFile_ren.pdb.
    
    sortRes or sR inpfile sortFile [outfile]
        Outputs the residues in the infile in the order specified in the sortFile and then 
        renumbers residues starting at 1. looks for inpFile or inpFile.pdb and if outfile is not specified
        writes to inpFile_sort.pdb
    
    removeDup or rd inpfile [outfile]
        Removes duplicate atoms by assessing the xyz position of each atom and makes sure it
        isn't output. Looks for inpFile or inpFile.pdb and if outfile is not specified
        writes to inpFile_noDup.pdb
    
    puckerGroupsRB or pgrb inpfile
        Outputs an rbodyconfig file containing the atoms of each hydroxyproline or proline in separate groups for
        each residue. 
              
    
    puckerGroupsRBS or pgrbs inpfile resfile
        Outputs an rbodyconfig file containing the atoms of each hydroxyproline or proline in separate groups for
        each residue specified in resFile 
    
    
    puckerGroups or pg inpfile OHflag [outfile]
        Finds prolines or hydroxprolines in the inpfile and outputs an atomgroups file containing
        the ring tip groups for rotating about the CD-CB axis for changing state between endo and exo.
        If OH flag is not zero also outputs the OH group rotations.
    
    puckerGroupSpec or pgs or PGS inpfile resfile OHFLag scaleFac propRot [outfile]
        Outputs an atomgroups file for the residues specified in resfile if they are HYP or PRO.
        Specifies the ring tip groups for rotating about the CD-CB axis for changing state 
        between endo and exo. If OH flag is not zero also outputs the OH group rotations.
        Scale Fac scales the rotation factor. probRot is the probability of a rotation occurring.
    
    checkTorsion or ct or cT or CT inpfile [outfile]
    
    torsionDiff or td or tD or Td or TD inpDirectory [outfile]
     
    residueInfo or RI or ri inpfile [outfile]
        takes a PDB and generates the per residue information such as torsionAngles, chi values and pucker status
    
    puckerBreakDown PBD or pbd inpfile
    takes pdb file and reports the break down of pucker states based on collagen Positions. i.e.
        Total Number of prolines+hyp, %age Exo, %age Endo
        Total Number of prolines in X Position,  %age Exo and %age Endo, overall contribution %age Exo and %age Endo,   
        Total Number of prolines in Y Position,  %age Exo and %age Endo, overall contribution %age Exo and %age Endo,
        Total Number of hypdroxyprolines in X Position, %age Exo and %age Endo, overall contribution %age Exo and %age Endo,
        Total Number of hypdroxyprolines in Y Position, %age Exo and %age Endo, overall contribution %age Exo and %age Endo,
    
    readSymmetry, 'rs', 'RS','rS','Rs' inpfile [capRes] [fig] [outfile]
        Returns the helical parameters of the collagen structure in the given pdb. Ignores capRes residues at either end of each chain in inpfile.
        FIgure plots the data from the fitting routine as we go. (slows things down Massively!). outfile specifies the output file name
            
    readResidueSymmetry','rrs','RRS']: inpfile [outfile]
        Fits and axis to all the CA, N and C atoms in all the residues. Then uses the axis to measure the helical parameters arising from each residue
        and the corresponding residue in the next GXY repeat. 
        Returns a list which is GXY units long. 
        Each list sets of three  [residue numbers, twist about the axis, Distance along axis, radius, number of units per period and true period].
        Dumps a gnuplot readable file.
            
    rotateGroup or RG or rg inpfile atomGroup [outfile]
        rotates all the atomgroups defined in the atomgroups file by angle specified in the atom groups file.
        This is a change to the conventional meaning of the parameter in the atomgroups file. The indicies are 
        order the atoms appear in the PDB.
        GROUP name atomaxis1 atomaxis2 groupsize angle
        atom1index
        atom2index 
    
    replacePdbXyz or xyz2pdb inpfile xyzfile [outputfile]
        takes the input pdb file and generates a new pdb file for every frame in the xyzfile
                        
"""
 
#source leaprc.ff99SB
#loadamberprep ../HYP/prepHYP05.in
#loadamberparams ../HYP/HYP05.forcemod
#mol=loadpdb 1V7H_noH.pdb
#saveamberparm mol 1V7H.prmtop 1V7H.inpcrd
#savepdb mol 1V7H_leap.pdb
#quit
    os.getcwd()
    infile=[]
    command=[]
    params=[]
    
    # process command line params
    if len(sys.argv) < 3:
        print usage
        sys.exit(1)
    else:
        command=sys.argv[1]
        infile=sys.argv[2]

    try:
        vst=open(infile,'r')
        vst.close()
    except:
        try:
            testfile=infile+'.pdb'
            vst=open(testfile,'r')
            infile=testfile
            vst.close()
        except:
            print "Unrecognised input file name: "+infile+" or "+infile+".pdb"
            exit(1)

    print "Command: " + command
    print "infile: " + infile

    # fragmentPDB or fpdb inpfile resfile
    if command in ['fragmentPDB', 'fragmentpdb', 'fpdb', 'fPDB', 'FPDB']:
        if len(sys.argv)==4:
            params=sys.argv[3]
            print "fpdb params: ", params
        else:
            print "fragmentPDB: Must specify inpfile and resflle:  fpdb inpfile resfile"
            exit(1)
            
    elif command in [ 'centrePDB', 'CPDB', 'cPDB', 'cpdb']:
        if len(sys.argv)!=3:
            print "cPDB: Must specify inpfile:  cPDB inpfile"
            exit(1)
            
    elif command in ['eliminateCIS', 'eCIS']:
        if len(sys.argv)!=3:
            print "eCIS: Must specify inpfile:   eCIS inpfile"
            exit(1)      

    elif command in ['generateCTAtomGroups', 'GCTAG']:
        if len(sys.argv)!=3:
            print "GCTAG: Must specify inpfile:   GCTAG inpfile"
            exit(1)      

    elif command in ['createCTFile', 'cCTF']:
        if len(sys.argv)==5:
            params=sys.argv[3:]
            print "createCTFile params: ", params
        else:
            print "createCTFile: Must specify inpfile, CTFile and threshold:   cCTF inpfile initCTFile thresh"
            exit(1)                  

    elif command in ['flipCT', 'fct']:
        if len(sys.argv)!=5:
            print "flipCT: Must specify inpfile, cistrans file and a force field:  fct inpfile cistran forcefield"
            exit(1)
        params = sys.argv[3:]         
    
    elif command in ['convertseq', 'ConvertSeq', 'cs', 'cS', 'CS', 'Cs']:
        if len(sys.argv)!=4:
            print "convertseq: Must specify inpfile and seq flle:  cs inpfile seqfile"
            exit(1)
        params = sys.argv[3]         
        
    # addTermini or at inpfile N(100,00) C
    elif command in ['addTermini', 'addtermini', 'AddTermini', 'at','aT','AT']:
        if (len(sys.argv)<3) or (len(sys.argv)>5):
            print "addTermini: Must specify inpfile and up to two termini (angles are optional). format N=a,b C=a,b "
            exit(1)
        if len(sys.argv)==4:            
            params = [ sys.argv[3] ]
        if len(sys.argv)==5:
            params = sys.argv[3:]
        print "add termini params: ", params
    
    # READ PUCKER
    elif command in ['readpucker', 'rp']:
        if len(sys.argv)==3:
            params = pl.fileRootFromInfile(infile)
            params = params+'.pucker'
        else:
            params=sys.argv[3]
        print "rp params: ", params

    # check pucker pattern
    elif command in ['checkPuckerPattern', 'cpp', 'CPP']:
        print 'No Params for check Pucker Pattern'
        # no params allowed for this command

    # Read Chirality
    elif command in ['readchirality', 'rc', 'RC','rC','Rc']:
        if len(sys.argv)==3:
            params = pl.fileRootFromInfile(infile)
            params = params + '.chi'
        else:
            params=sys.argv[3]
        print "rc params: " + params

    # group atoms
    elif command in ['groupAtomsXYZ', 'gAX', 'gax', 'GAX']:
        if len(sys.argv)==3:
            params = pl.fileRootFromInfile(infile)
            params = params + '.xyz'
        else:
            params = sys.argv[3]
        print "gax params: " + params

    # make xyz for blender
    elif command in ['makeXYZForBlender', 'mXB', 'mxb']:
        fileroot = pl.fileRootFromInfile(infile)
        outfile = fileroot + '.xyz'
        if len(sys.argv)==3:
            # backbone only, give each residue a unique name 
            params = [1, 2, outfile]
        elif len(sys.argv)==5:
            params = [ int(sys.argv[3]), int(sys.argv[4]), outfile ]
        else:
            print usage
            sys.exit()
        print "mXB params: ", params

    # Fix Chirality
    elif command in ['fixchirality', 'fc', 'FC','fC','Fc']:
        if len(sys.argv)==3:
            params = pl.fileRootFromInfile(infile)
            params = params + '_chi.pdb'
        else:
            params = sys.argv[3]
        print "fc params: " + params

    # ramachandran 
    elif command in ['ramachandran', 'rmc']:
        params = [ pl.fileRootFromInfile(infile)+'.gplt' ]
        params.append(pl.fileRootFromInfile(infile)+'.rama')
        
        print "ramachandran plot command not implemented: ", params

    #replace pdb with xyz data
    elif command in ['replacePdbXyz','xyz2pdb']:
        if len(sys.argv)<4:
            print "ReplacePdbXyz usage: pdbproc.py xyz2pdb inpfile xyzfile [outfile]"
            sys.exit(0)
        elif (len(sys.argv)==4):
            params=[sys.argv[3]]
            params.append(pl.fileRootFromInfile(infile))
        elif (len(sys.argv)>4):
            params=sys.argv[3:5]

        print "xyz2pdb command: "
        print params

    #Rotate groups
    elif command in ['rotateGroup', 'RG', 'rg']:
        if len(sys.argv)<4:
            print "Rotate group usage: pdbproc.py rotateGroup inpfile atomgroups [outfile]"
            sys.exit(0)
        elif (len(sys.argv)==4):
            params = [ sys.argv[3] ]
            params.append( pl.fileRootFromInfile(infile) + '.rot.pdb' )
        elif (len(sys.argv)>4):
            params = sys.argv[3:5]

        print "rotateGroup command: "
        print params

    ##Read Symmetry
    elif command in ['readSymmetry', 'rs', 'RS','rS','Rs']:
        if len(sys.argv)==3:
            params=[0]
            params.append(0)
            params.append( pl.fileRootFromInfile(infile) + '.sym' )
            
        else:
            if len(sys.argv)==4:
                params=[int(sys.argv[3])]
                params.append( 0 )
                params.append( pl.fileRootFromInfile(infile) + '.sym' )
                    
            else:
                if len(sys.argv)==5:
                    params = [int( sys.argv[3] ) ]
                    params.append( sys.argv[4] )
                    params.append( pl.fileRootFromInfile(infile) + '.sym' )
                else:
                    #6 params or longer longer... 
                    params=[int( sys.argv[3]) ]
                    params.append( sys.argv[4] )
                    params.append( sys.argv[5] )

                    
        print ["read symmetry params: "] 
        print params

    ##Pucker State Summary; has no command parameter processing; just has an infile
    elif command in ['puckerBreakDown', 'PBD', 'pbs']:
        print "pucker break down invoked"
 
    #### Convert Proline to HydroxyProline
    elif command in ['proToHyp','PH','pH','Ph','ph']:
        if len(sys.argv)<4:
            print "proToHyp: you must specify an input pdb file and a file with a list of residue to convert, and an optional output file"
            print usage
            sys.exit(1)
        elif len(sys.argv)==5:
            params = sys.argv[2:]
        else:
            params.append( sys.argv[3] )
            params.append( pl.fileRootFromInfile(infile) + '_hyp.pdb' )


    #### read Sequence
    elif command in ['readSequence','rsq','RSQ']:
        if len(sys.argv)<4:
            print "readSequence input.pdb mode [width]"
            exit(1)
        elif len(sys.argv)==4: 
            params=[sys.argv[3]]
        else:
            params=sys.argv[3:]
            
        params.append( pl.fileRootFromInfile(infile) + '.seq' )
        l = "readSequence params: "
        for p in params:
            l = l + str(p) + ' '
        print l

    #### Replace Sequence
    elif command in ['modifySequence','MS','mS','Ms','ms']:
        if len(sys.argv)<5:
            print "modifySequence input.pdb newSeq resNumStart [output.pdb]"
            print usage
            exit(1)
        elif len(sys.argv)==6:
            params=sys.argv[3:]
        else:
            params = [ sys.argv[3:] ]
            params.append( pl.fileRootFromInfile(infile) + '_ms.pdb' )

    #### RENAME TERMINI
    elif command in ['renameTermini','rt','rT','RT']:
        if len(sys.argv)<4:
            print "renameTermini: you must specify an input pdb file, and a flag"
            print usage
            exit(1)
        elif len(sys.argv)==5:
            params = sys.argv[2:]
        else:
            params.append( sys.argv[5] )
            params.append( pl.fileRootFromInfile(infile) + '.prmtop' )

    #### renumberRes or RN
    elif command in ['renumberRes','rn','rN','RN']:
        #check for an outputfile
        if len(sys.argv)==3:
            params.append( pl.fileRootFromInfile(infile) + '_ren.pdb' )
        elif len(sys.argv)==4:
            params.append(sys.argv[3])
            params.append( pl.fileRootFromInfile(infile)+'_ren.pdb' )
        elif len(sys.argv)==5:
            params.append(sys.argv[3])
            params.append(sys.argv[4])
        print 'renumberRes', params

    #### removeDuplicates
    elif command in ['removeDup','rd','RD','rD']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append(sys.argv[3])
        else:
            params.append( pl.fileRootFromInfile(infile) + '_noDup.pdb' )
        print 'removeDup', params

    #### Read Residue Symmetry
    elif command in ['readResidueSymmetry','rrs','RRS']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append(sys.argv[3])
        else:
            params.append( pl.fileRootFromInfile(infile) + '.resSym' )
        print 'removeDup', params

    #### checkTorsion
    elif command in ['checkTorsion','ct','cT','Ct','CT']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append( sys.argv[3] )
        else:
            params.append( pl.fileRootFromInfile(infile) + '.torsion' )
        print 'checkTorsion', params

    #### residueInfo    
    elif command in ['residueInfo','RI','Ri','rI','ri']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append( sys.argv[3] )
        else:
            params.append( pl.fileRootFromInfile(infile) + '.residue' )
        print 'residueInfo', params


    #### torsionDiff
    elif command in ['torsionDiff','td','tD','Td','TD']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append(sys.argv[3])
        else:
            params.append('diff.torsion')
        print 'torsionDiff', params

    #### puckerGroupsRigidBody
    elif command in ['puckerGroupsRB','pgrb']:
        #no output file or flags to specify.
        params.append('rbodyconfig')

    #### puckerGroupsRigidBodys
    elif command in ['puckerGroupsRBS','pgrbs']:
        #check for an outputfile
        if len(sys.argv)<3:
            print "Usage: puckerGroupsRBS <inpfile.pdb> <resnumbersFile>"
            print usage
            exit(1)
        else:
            params.append(sys.argv[3])
            params.append('rbodyconfig')

    #### puckerGroups
    elif command in ['puckergroups','pg','PG','Pg','PG']:
        #check for an OH flag and an output file
        if len(sys.argv)>3:
            if len(sys.argv)==4:
                params.append(sys.argv[3])
                params.append('atomgroups')
            elif len(sys.argv)>4:
                params.append(sys.argv[3])
                params.append(sys.argv[4])
        else:
            params.append(int(0))
            params.append('atomgroups')

    #### puckerGroupSpec
    elif command in ['puckergroupSpec','pgs','PGS']:
        #check for an outputfile
        if len(sys.argv)<7:
            print "Usage: puckergroupSpec <inpfile.pdb> <resnumbersFile> <OHFlag> <scaleFac> <probRot> [<outputfile>]"
            print usage
            exit(1)
        else:
            params=sys.argv[3:]
            if len(sys.argv)==7:
                params.append('atomgroups')

    #### PREP FOR AMBERGMIN
    elif command in ['prepAmberGmin','pag','PAG','pAG']:
        if len(sys.argv)<5:
            print "prepAmberGMIN: you must specify an input file, a rule file and a forcefield."
            print usage
            exit(1)
        else:
            params=sys.argv[3:]

    #### SORT RESIDUES
    elif command in ['sortRes', 'sr', 'sR' ,'SR']:
        if len(sys.argv)<4:
            print "sortRes: you must specify an input PDB file and a sortfile."
            print usage
            exit(1)
        else:
            params.append(sys.argv[3])
            try:
                vst = open(params[0],'r')
                vst.close()
            except:
                try:
                    testfile = params[0] + '.sort'
                    vst = open(testfile,'r')
                    params[0] = testfile
                    vst.close()
                except:
                    print "Unrecognised input filename:" + params[0] + " or " + params[0] + ".sort"
                    exit(1)

        if len(sys.argv)>3:
            params.append( pl.fileRootFromInfile(infile) + '_sort.pdb' )
        else:
            params=sys.argv[4]

        print "sortResidue params: "
        print params

    #### REMOVE ATOMS    
    elif command in ['removeatoms', 'ra']:
        if len(sys.argv)<4:
            print "removeatoms: you must specify a rule file and an input PDB file."
            print usage
            exit(1)
        else:
            params.append(sys.argv[3])
            try:
                vst=open(params[0],'r')
                vst.close()
            except:
                try:
                    testfile=params[0]+'.lines'
                    vst=open(testfile,'r')
                    params[0]=testfile
                    vst.close()
                except:
                    print "Unrecognised input filename:"+params[0]+" or "+params[0]+".lines"
                    exit(1)
        if len(sys.argv)==4:
            params.append( pl.fileRootFromInfile(infile) + '_short.pdb' )
        else:
            params=sys.argv[4]
            print "ra params: "
            print params
 
    #### WRITE PUCKER
    elif command in ['writepucker', 'wp']:
        if len(sys.argv)<4:
            print "Must specify a pucker state file and an input PDB file."
            print usage
            exit(1)
        else:
            #mandatory pucker filename
            params.append(sys.argv[3])

        #validate pucker filename
        try:
                vst=open(params[0],'r')
                vst.close()
        except:
            try:
                testfile = params[0] + '.pucker'
                vst = open(testfile,'r')
                params[0]=testfile
                vst.close()
            except:
                print "Unrecognised input file name: " + params[0] + " or " + params[0] + ".pucker"
                exit(1)
                
        #optional outputfilename
        if len(sys.argv)==4:
            params.append( pl.fileRootFromInfile(infile)+'.pucker.pdb' )
        else:
            params.append(sys.argv[4])

        print "wp params: "
        print params
    else:
        print "Unrecognised Command"
        exit(1)
        
    return [infile, command, params]

if __name__ == '__main__':

        [infile, command, params] = commandLineProc()

        if command in ['readpucker','rp']:
            pl.readpucker(infile,params)
        
        elif command in ['fragmentPDB', 'fragmentpdb', 'fpdb', 'fPDB', 'FPDB']:
            pl.fragmentPDB(infile, params)
        
        elif command in [ 'centrePDB', 'CPDB', 'cPDB', 'cpdb']:
            pl.centrePDB(infile)
        
        elif command in ['generateCTAtomGroups', 'GCTAG']:
            pl.generateCTAtomGroups(infile)

        elif command in ['eliminateCIS', 'eCIS']:
            pl.eliminateCIS(infile)

        elif command in ['createCTFile', 'cCTF']:
            pl.createCTFile(infile, params)

        elif command in ['addTermini', 'addtermini', 'AddTermini', 'at','aT','AT']:
            pl.addTermini(infile, params)            
        
        elif command in ['convertseq', 'ConvertSeq', 'cs', 'cS', 'CS', 'Cs']:
            pl.convertSequence(infile, params)        
        
        elif command in ['readchirality','rc','RC','rC','RC']:
            pl.readchirality(infile,params)
        
        elif command in ['groupAtomsXYZ', 'gAX', 'gax', 'GAX']:
            pl.groupAtomsXYZ(infile, params)
        
        elif command in ['makeXYZForBlender','mXB','mxb']:
            pl.makeXYZForBlender(infile, params[0], params[1], params[2])
        
        elif command in ['fixchirality','fc','FC','fC','FC']:
            pl.fixchirality(infile, params)
        
        elif command in ['writepucker','wp']:
            pl.writepucker(infile, params)
        
        elif command in ['removeatoms','ra']:
            pl.removeLineList(infile, params)
        
        elif command in ['prepAmberGmin','PAG','pAG','pag']:
            pl.prepAmberGMin(infile, params)
        
        elif command in ['flipCT', 'fct']:
            pl.flipCT(infile, params)
        
        elif command in ['renameTermini','rt','rT','RT']:
            pl.renameTerminiTop(params[1],infile, int(params[0]))
        
        elif command in ['renumberRes','rn','rN','RN']:
            pl.renumberResidues(infile,params[0],params[1])
        
        elif command in ['sortRes','sr','sR','SR']:
            pl.sortResidues(infile,params[0],params[1])
        
        elif command in ['proToHyp','ph','pH','Ph','PH']:
            pl.proToHyp(infile,params[0],params[1])
        
        elif command in ['removeDup','rd','rD','Rd','RD']:
            pl.removeDuplicateAtoms(infile,params[0])
        
        elif command in ['checkTorsion','ct','cT','Ct','CT']:
            pl.checkTorsion(infile,params[0])
        
        elif command in ['torsionDiff','td','tD','Td','TD']:
            pl.torsionDiff(infile,params[0])
        
        elif command in ['modifySequence','ms','mS','Ms','MS']:
            pl.modifySequence(infile,params[0],params[1],params[2])
        
        elif command in ['readSequence','rsq','RSQ']:
            if len(params)==2:
                pl.readSequence(infile, int(params[0]), params[1], width=80)
            else:
                pl.readSequence(infile, int(params[0]), params[2], width=int(params[1]))
            
        elif command in ['puckergroups','pg','pG','Pg','PG']:
            pl.puckerGroups(infile,params[0],params[1])
        
        elif command in ['puckergroupSpec','pgs','PGS']:
            pl.puckerGroupSpec(infile,params[0],params[1],params[2],params[3],params[4])
        
        elif command in ['puckerGroupsRB','pgrb']:
            pl.puckerGroupsRB(infile,params[0])
        
        elif command in ['puckerGroupsRBS','pgrbs']:
            pl.puckerGroupsRBs(infile,params[0],params[1])
        
        elif command in ['checkPuckerPattern','cpp','CPP']:
            pl.checkPuckerPatternWrapper(infile)
        
        elif command in ['residueInfo','RI','Ri','rI','ri']:
            pl.residueInfo(infile,params[0])
        
        elif command in ['puckerBreakDown','PBD','pbd']:
            pl.puckerBreakDown(infile)           
        
        elif command in ['readSymmetry','rs','RS','rS','Rs']:
            pl.readSymmetry(infile,int(params[0]),int(params[1]),params[2])
        
        elif command in ['readResidueSymmetry','rrs','RRS']:
            pl.readResidueSymmetryWrapper(infile,params[0])
        
        elif command in ['rotateGroup', 'RG', 'rg']:
            pl.rotateGroup(infile,params[0],params[1])
        
        elif command in ['replacePdbXyz', 'xyz2pdb']:
            pl.replacePdbXYZ(infile,params[0],params[1])
        else:
            print "Unknown Command: " + command
