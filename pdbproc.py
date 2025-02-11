#!/usr/bin/env python
import sys
import os
import pdbLib as pl
import json

def commandLineProc():

    usage="""
pdb file processor
==================

General Syntax:	pdbproc.py command inpfile1 [inpfile2] [param1] [param2]

where command and inpfile are mandatory. Command is the action to perform and inpfile
is the name of a PDB file in standard format. Each command takes variable number of parameters.

command is one of:

    addTermini inpfile N=-30,10 C
        aliases: addTermini, addtermini, AddTermini, at, aT, AT 
        Adds N or C terminus line to PDB (NME, ACE) if N and/or C are present in the params. 
        A dihedral and bond angle pair may be specified for each terminus. 
        otherwise default is 0, 109.
        Bond length is difference between C and CA or CA and N in adjacent residue. 

    alignPdbs inpfile1 inpfile2 config.json
    
        takes two PDBs and aligns the second with the first returning a third PDB with the aligned structure.
        parmeters for json:
        
        {
          'backboneOnly': True,
          'position': 1
        }
            
        backboneOnly specifies that only the backbone are used to perform the alignment and the position specifies 
        which residues in the first structure should be aligned with those in the second structure. 
    
        see also correlate PDBs, which runs this algorithm at each point


    backboneAtomGroups inpfile angRange
        
        aliases: BBAG, bbag
                
        Creates an atom groups file with groups from each CA to the end of the chain and 
        also from each N to the end of the chain. This enables exploration of each phi and psi
        angle "independently".  Intended to be a way to quickly collapse a straight chain into far apart 
        structures.  After a certain point can be followed up with side chain moves as the 
        chain collapses. AngRange determines the size range of the rotations. (-180 to 180 degrees. 
        Converted to radians internally.)  

    centrePDB CPDB or cPDB inpfile
        translates the PDB to it's centre of mass frame and outputs identical pdb with only coords modified.

    checkPuckerPattern or cpp inpfile 
        prints true or false if the pdb file contains a ring pucker system
        whose endo and exo states match the endo and exo states of the standard collagen fibre.
    
    checkTorsion or ct or cT or CT inpfile [outfile]
   
    concatenatePDBS cpdbs inpfile configfile
        Takes a glob of PDBS (typically different conformations of the same structure) and then adds them as different models in a single file. 
        Useful for making movies in VMD. 
   
    correlatePdbs or corrpdb inpfile1 inpfile2
        Takes two PDBs and aligns the second at each residue position of the first.  
        Essentially performs an "alignment correlation" along their structure length using the Kabsch alignment algorithm in Scipy.
        
        A file is dumped containing the RMSD information as well as the hamming distance of the short sequence 
        from the local sequence in the first structure.
        
        A figure is produced showing the rmsd correlation function and hamming distance 

        see also alignPDBs which uses the same internal alignment algorithm and dumps a pdb of the second file aligned at a specific position. 
   
    crankShaftGroups, crankshaftgroups, csg inpfile nnCutoff dCutoff rotScale include sidegroups 
        Take a pdb and generates an atomgroups file for the GMIN rotategroups command which defines a set of rotation groups 
        consisting of the intervening chain between pairs of carbon alphas.
        
        All possible pairs of CAs in the pdb are chosen that satisfy both the nnCutOff and dCutoff constraints. 
        
        nnCutoff is the nearest neighbour cutoff and specifies the max number of aminoacids along the backbone between the chosen CAs.
            Eg. nnCutoff=10 will iterate through each 10 residue sub-sequence on each chain to produce pairwise 
            combinations of all pair of CAs in each 10 residue sub-sequence. 10-9, 10-8 and so on without reps (8-10, 9-10) etc. 
        
        dCutoff will find all CAs that are within euclidean distance dCutoff of each CA and produce pair wise combinations 
            of CAs within that spatially bound region.  
            
        The final list of pairs of CA is checked and any pairs whose distance along the chain is > nnCutoff will 
            be eliminated.  And pairs of CAs < nnCutoff that are more than dCutoff away will also be eliminated. 

        rotScale determines the rotationScale that each particular cranks shaft rotation will be selected as percentage of 2*pi:w
            
        Include is flag that is true (1) or false (0) and specifies whether or not the side chain on the CA is included in the 
        rotation group or not. Default: 0.   
           
        sidegroups is a flag that specifies whether or not the sidechains of ALL the amino acids are included as
        atomgroups for rotation.  All possible rotatable subgroups of each sidechain are specified as rotation groups.
        For prolines or closed rings, the end group is flipped as in a pucker flip. Default: 0    
        
   
    convertseq, ConvertSeq, cs, cS, CS, Cs inpfile seqfile
        Takes an input file which consists of single letter codes of a protein sequence, like a mode 3 sequence file 
        from readSequence. Outputs a file with filename seqfile which has the three letter codes as per mode 2 of readSequence.
        The three letter codes output file is compatible with the modify sequence command.  

    createCTFile, cCTF inpfile thresh
        generates a CIS_TRANS_STATE file for a given PDB

    createSequence, CSQ inpfile type term
        Uses the vesiform functionality to create a sequence. 
         
        inpfile is a type 3 sequence file. Generates a backbone N C CA according to the type.
        
            type is alpha, beta, or straight. 
            
            This defines the dihedral angles phi and psi of the backbone. Omega is always 180.
        
            Once the back bone is generated, a partial PDB is output that has the N CA, and C coords and the sequence
            names of the residues.   This can be fed into tleap to populate the rest of structure and generate amber params.
            
            If term is 1 then ACE and NME are added also. 
                
    dumpParticularAtoms, dpa inpfile config.json
        Dumps specific atoms in PDB format.  
        A parameter in the json file 'atomsToDump' is a list of atoms names to dump e.g.
         {
             "atomsToDump": [ 'CA', 'C','N'], 
             "outfile": 'filename' 
         }
         
         will dump the backbone (CA, C and N of the pdb into a fresh pdb called "filename".
         If outfile name is not present dumps as e.g. infileRoot_CA_C_N.pdb

    dumpCAsAsPDB dCA infile 
        dumps the input pdb containing only the CAs of a protein. renames as infile_CAOnly.pdb
         
    eliminateCIS eCIS inpfile
	    Calls the createCT algorithm to check if the lowest energy pdb has no CIS states
	    if it does then it generates a new atomGroups file to rotate each half of the CIS peptide bonds by pi/2.
        Then calls CUDAGMIN for a bunch of steps with CISTRANS checking turned off. 
	    Loops until the lowest energy minimum has no CIS peptide bonds

    fixChirality or fc inpfile [outfile]
        Checks the chirality state of each proline and hydroxyproline  
        in the inpfile. Output is resNum with an 2-X or 4-X where x is R or S.
        If the chirality is not a 2-S 4-R state then it is converted to be so
        and rewritten as a PDB. If not specified the output filename is ..._chi.pdb
        
        See readChirality command

    flipCT fct inpfile cis_trans forcefield
        Takes a cis trans file from gmin, and performs a series of rotations to convert all the cis peptides to trans.
        Then runs it through the PAG process with a noreduce optionto generate a restart file

    fragmentPDB, fragmentpdb, fpdb, fPDB, or FPDB inpfile resfile 
    
        Generates a series of minipdbs based on the pairs of start and stop residues in resfile.
        if the start residues does not exist then the output starts at the first residue.
        If the end residue does not exist the tail of the PDB is provided from the start residue onwards.
        iF the end residue is before the start residue then no output is returned. 
        
        example resfile
        1 10
        1 100
        35 39
   
    frenetFrameAnalysis ffa inpfile.pdb config.json
        computes the frenet frame at each point on the back bone of the given PDB and 
        determines the curvature and torsion of the backbone at each residue.
        dumps a straight up text file with K and T at each residue and plots a graph with parameters defined in config.json 
   
   
    generateCTAtomGroups, GCTAG infile
        takes a initial_cis_trans_state file and creates an atomgroups file for all the CIS peptide bonds. Uses information
        in the pdb file pointed at by infile to help construct the atomgroups file

    generateFreezeDryGroups gfg infile resIdRBody1 resIdRBody2 angRange sidechains
    
        generates a rigidbody file and atomgroups files for the freezeDry process. 
        
        for the given pdb specifies:
           a rigid body which goes from the NTerm to the last carbon of amino acid with resId resIdRBody1
           a rigid body which goes from the Nitrogen of amino acid with resId resIdRBody2 to the CTerm
           a sets up an atomsgroup file which has groups from the CA and N of each amino acid between 
           resIdRBody1 and resIdRBody2 to the CTerm.
           specify the angRange of rotation for each group rotation.
           Decide if the side groups of the amino acids between resIdRBody1 and resIdRBody2 should be included in the atomsgroups file
            

    groupAtomsXYZ, gAX or gax inpfile
        reads the PDB and dumps the atoms XYZ file, but relabels the atom types according 
        to basic atom type. All Cs are called C, all H1, HE1 etc become H.  Reduces number of objects
    loaded into blender

    gyrationAnalysis gyrA inpfile, params
        computes the radius of gyration of a molecule about its principal moments of inertia
        and for the molecule as a whole
        returns the extent of the object in each of those principal directions  

    GanTrain gc trainSetDirectory configs.json
    
        takes a training set of PDBs and creates a generator and discriminator GAN model and saves it in the input dir.
        Does some validation between a random set of input distances and a reserved part of the training set.
        

    GanCompare trainSetDirectory testSetDirectory configs.json

        Loads a model from the trainingSet directory and the data from the testSet Directory
        and outputs the what the trained discriminator model thinks regarding the nature of the testSet

        Collates a score for every NMer in a file and returns an aggregate score for the file. 
        Also compiles an aggregate score for all the files against that Gan Model.

    makeXYZforBlender ir mXB inpfile backboneOnly mode
        reads the PDB and creates and xyz for reading into blender. If backboneOnly is 1, then it dumps 
        the CA residue as an XYZ file. Atoms in each residues labelled according to the mode. 
        1 = functional nature of the amino acid that the CA belongs to which generates a different colour for each type 
        of residue 
        2 = Each of the twenty residues and IMB for PLPs -> A unique atom. 

    measureTubeRadius mtr inpfile config.json
        Takes a pdb, loads it in and and finds the CAs. Takes the COM average of n residues and sweeps along the structure in steps of m.
        approximates a line of best fit to each m long section of the COM axis. 
        Computes the distance of each residue in each section from the axis and counts how many residues are in each cylindrical bin creating a sigmoidal plot.
        Plots all the sigmoidal plots along the whole system on top of each other.
        Plots the distance of each residue against the residue number and colors each resiude.     
    
    

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

        see readSequence Command
    
    multimerRg mrg inpfile config.json
    
        Places multiple copies of a pdb structure close to each other and computes the Rg of the combined structure.
        The first PDB is centered at the origin without reorientation. Must define the relative position and orientation 
        of each copy in the lab frame. There's no checking for overlap. It just blindly follows the translation and then dumps
        the combined construct for inspection.
         
    
    atomicpairwisedistance apwd inpPDB configfile.json
        Computes the contact distance between all atoms in a PDB and returns a histogram of the distances. This is the 
        quantity that when fourier transformed gives the SAXS intensity profile.  You could inverse transform the SACX profile
        and the get the same thing.  
       
    
    pairwisedistance pwd inpPDB configfile.json
        Computes the contact distance between residues. Takes a PDB or a glob defined in the configfile and uses various algorithms to 
        define sets of residues to find the distance between them. Plots a matrix of the distance of Center of mass of all CAs in all 
        specified sets.
    
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

    puckerBreakDown PBD or pbd inpfile
        takes pdb file and reports the break down of pucker states based on collagen Positions. i.e.
            Total Number of prolines+hyp, %age Exo, %age Endo
            Total Number of prolines in X Position,  %age Exo and %age Endo, overall contribution %age Exo and %age Endo,   
            Total Number of prolines in Y Position,  %age Exo and %age Endo, overall contribution %age Exo and %age Endo,
            Total Number of hypdroxyprolines in X Position, %age Exo and %age Endo, overall contribution %age Exo and %age Endo,
            Total Number of hypdroxyprolines in Y Position, %age Exo and %age Endo, overall contribution %age Exo and %age Endo,
    
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

    RadiusOfGyration rg inpfile
        computes radius of gyration of a pdb structure
        
    ramachandran or rmc inpfile, configFile
        Makes a ramachandran plot of the pdb file.  the plot settings are controlled via the configFilej JSON
    
    ramachandrans or rmcs inpfile, inpdir, configFile
        Makes a ramachandran plot of each file in a glob defined in the inpDir.
        ignores inpFile, but does test that it exists for backwards compatibility
        The plot settings are controlled via the configFile in json format as for ramachandran
    
    ramadensity or rmd inpfile, configFile
        Makes a 2D Histogram of the phi and psi angles in the pdb file.  The plot settings are controlled via the configFilej JSON
        
    readChirality or rc inpfile [outfile]
        Checks the chirality state of each proline and hydroxyproline  
        in the inpfile. Output is resNum with an 2-X or 4-X where x is R or S.
        Generally should be an 2-S 4-R state.  if it is proline then just get a 2-R state.
        If not specified the outfile is a .chi file. See fixChirality command

    readpucker or rp inpfile [outfile]
        Checks the pucker state of each residue in inpfile.
        looks for inpfile (automatically appends .pdb if it is not there) and
        analyses the pucker state of each residue before outputting data into
        outfile in a format of resId, resName, Endo|Exo. 
        If outfile is omitted readpucker outputs data in inpfile.pucker.
        
        see writepucker command

    readResidueSymmetry','rrs','RRS']: inpfile [outfile]
        Fits and axis to all the CA, N and C atoms in all the residues. Then uses the axis to measure the helical parameters arising from each residue
        and the corresponding residue in the next GXY repeat. 
        Returns a list which is GXY units long. 
        Each list sets of three  [residue numbers, twist about the axis, Distance along axis, radius, number of units per period and true period].
        Dumps a gnuplot readable file.

    readSequence or rsq,RSQ  inpDB mode [width] 
        reads the sequence from the PDB and outputs into file with a .seq on the end.
        Enables easy editing of the PDB sequence. 
            Mode = 1 yields a numbered list of residues. 
            Mode = 2 yields a list of three letter residue codes in a format suitable for modifySequence.
            Mode = 3 yields a string of single chars representing each residue with the standard letters.
            Mode = 4 is three letter codes separated by spaces. 
            Mode = 5 uses an optional width parameter to output 1 letter codes in lines width chars long 
                     as per fasta file format. Defaults to 80 if omitted.
                     
            see modifySequence command. 
    
    readSymmetry, 'rs', 'RS','rS','Rs' inpfile [capRes] [fig] [outfile]
        Returns the helical parameters of the collagen structure in the given pdb. Ignores capRes residues at either end of each chain in inpfile.
        FIgure plots the data from the fitting routine as we go. (slows things down Massively!). outfile specifies the output file name

    removeAtoms or ra inpfile rulefile [outfile]
        loads a pdb file and removes lines which match the specifications and then rewrites the pdb file. 
        looks first for infile or infile.pdb and loads it. 
        Then looks for the rulefile or rulefile.lines. which has the format:

        string integer
    
        where string is the string which is contained in the ATOM line that qualifies it for removal, 
        eg H, C, GLY or whatever.  integer is the zero based column number in which the text is to appear.
        envisaged for use solely with the ATOM keyword. 
        The rest of the PDB file is output verbatim, except for the lines which qualify.
        if the outfile is not specified then the default is infile_short.pdb.

    removeDup or rd inpfile [outfile]
        Removes duplicate atoms by assessing the xyz position of each atom and makes sure it
        isn't output. Looks for inpFile or inpFile.pdb and if outfile is not specified
        writes to inpFile_noDup.pdb
    
    renameTermini or rT inpfile flag [topologyfile]
        looks for the pdb inpfile or inpfile.pdb and loads it. Then it scans for inpfile.prmtop
        or topologyFile and then adds or removes Ns and Cs to the N and C termini of each chain
        in the topologyFile. The topologyFile is overwritten. If Flag is 0 the topology file is
        output without the Ns and Cs. If Flag is 1 the topology file is output with the Ns and Cs.
    
    renumberRes or RN inpfile [startNum] [outfile]
        starts numbering residues at the beginning of a file and everytime it finds a Nitrogen increments
        the residue number. outputs using standard rules.  looksfor inpFile or inpFile.pdb and if outfile
        is not specified writes to inpFile_ren.pdb.

    replacePdbXyz or xyz2pdb inpfile xyzfile [outputfile]
        takes the input pdb file and generates a new pdb file for every frame in the xyzfile
       
    residueInfo or RI or ri inpfile [outfile]
        takes a PDB and generates the per residue information such as torsionAngles, chi values and pucker status

    rotateGroup or RG or rg inpfile atomGroup [outfile]
        rotates all the atomgroups defined in the atomgroups file by angle specified in the atom groups file.
        This is a change to the conventional meaning of the parameter in the atomgroups file. The indicies are 
        order the atoms appear in the PDB.
        GROUP name atomaxis1 atomaxis2 groupsize angle
        atom1index
        atom2index 
    
    scanSetForSequence ssfs inpfile inputDirectory outputDirectory
    
        loads an inpfile, gets the sequence from it and then scans a directory of pdbs for any pdbs that contain 
        that sequence. Outputs the segment of each pdbs that matches the sequence perfectly as a new pdb with the 
        same file name that it came from plus the first 10 letters of the sequence.
    
    solvExposure solE inpfile config.json 
    
        Loads in input file and submits the pdb to a website for solvent based analysis, returning a report 
        on which residues are solvent exposed.
    
    sortRes or sR inpfile sortFile [outfile]
        Outputs the residues in the infile in the order specified in the sortFile and then 
        renumbers residues starting at 1. looks for inpFile or inpFile.pdb and if outfile is not specified
        writes to inpFile_sort.pdb
    
    spherowrap or sw inpfile config.json
    
        creates a minimal segmented spherocylinder as an envelope around the collection of points stored in the pdb file. 
    
    surfaceAnalysisComplete sac inpfile config.json
        Performs both tubule and GB analysis of a PDB and outputs a complete summary, including a plot of contour length 
        with different averages and intervals for a given PDB. 
    
    
    surfaceFindGB sfg inpfile config.json 
    
        Analyses a pdb and finds surface of protein, 
        Plots analysis of where residues are relative to that surface. 


    surfaceFindTube sft inpfile config.json

        Computes the center of mass of subsets of amino acids a certain distance apart along a protein chain. 
        Generates a smoothed version of the "fold" for a PDB. The vector between each center of mass defines an axis.
        Compute the distance of each residue (CA) from every segment. Assign each residue to its closest segment.  Use that 
        to compute the distance of each segment from the notional axis. 
    
        Create a histogram of the distance of each residue type from the axis.
    
        Gets a value for the contour length of the spidroin tube. 
        
    Surfaces sfs inpfile config.json
    
        for every model in a PDB run the surfaces algorithm and compute a histogram. Enables the center of mass 
        and tube radius to emerge over time.

        
    torsionDiff or td or tD or Td or TD inpDirectory [outfile]


    trajectoryCluster or tc referenceFile inpDirectory configFile
    
        takes a directory of pdbs and builds a model categorises every nmer backbone sequence to determine different types of turns etc.
        
        Must align every nmer subsequence with a reference structure first. 
        
        Creates a finger print of a set of proteins.  
    
    trajectoryMeasure or tm inpfile modelDirectory configFile
        
        Classifies a protein according to its score against each classes of nmer category in a kind of vector score for the protein.
        We can take each nmer sub chain and project it into a category, and then sum how many of each category go to make up the whole protein. 
    

    writepucker or wp inpfile puckerStateFile [outfile]
        sets the pucker state of each residue in the inpfile pdb to the status specified in puckerStateFile.
        The pucker State file contains lines with the format resiD, resName, Endo|Exo.
        Looks for inpfile (automatically appends .pdb if it is not there).
        A pucker inversion operation is applied to prolines or hydroxyprolines which do not match the
        required state. Outfile is an optional outputfilename. Outfile is omitted the output file name 
        is inpFileRoot.pucker.pdb.
        
        see readpucker command
                        
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
        print(usage)
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
            print("Unrecognised input file name: "+infile+" or "+infile+".pdb")
            exit(1)

    print("Command: " + command)
    print("infile: " + infile)

    # fragmentPDB or fpdb inpfile resfile
    if command in ['fragmentPDB', 'fragmentpdb', 'fpdb', 'fPDB', 'FPDB']:
        if len(sys.argv)==4:
            params=sys.argv[3]
            print("fpdb params: ", params)
        else:
            print("fragmentPDB: Must specify inpfile and resflle:  fpdb inpfile resfile")
            exit(1)

    elif command in ['alignPdbs', 'apdb']:
        if len(sys.argv)==5:
            with open(sys.argv[4], "r") as f: 
                params=json.load(f)
            params['inpfile1']=sys.argv[2]
            params['inpfile2']=sys.argv[3]
            print("alignpdb params: ", params)
        else:
            print("alignPdbs: Must specify inpfile1, inpfile2 and configfile:  alignPdbs inpfile1 inpfile2 config.json")
            exit(1)

    elif command in ['correlatePdbs', 'corrpdb']:
        if len(sys.argv)==5:
            with open(sys.argv[4], "r") as f: 
                params=json.load(f)
            params['inpfile1']=sys.argv[2]
            params['inpfile2']=sys.argv[3]
            print("correlatePdbs params: ", params)
        else:
            print("correlatePdbs: Must specify inpfile1, inpfile2 configfile.json")
            exit(1)

    elif command in ['frenetFrameAnalysis', 'ffa']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("frenetFrame Analysis: inpfile:", infile, ", params: ", params)
        else:
            print("frenetFrameAnalysis: Must specify infile configfile.json")
            exit(1)

    elif command in ['dumpParticularAtoms','dpa']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("dpa params: ", params)
        else:
            print("dumpParticularAtoms: Must specify inpfile and configfile:  dap inpfile config.json")
            exit(1)

    elif command in ['measureTubeRadius','mtr']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("mtr params: ", params)
        else:
            print("measure tube radius : Must specify inpfile and configfile:  mtr inpfile config.json")
            exit(1)


    elif command in ['GanTrain', 'gt']:
        if len(sys.argv)==5:
            with open(sys.argv[4], "r") as f: 
                params=json.load(f)
            params['trainSetDir'] = sys.argv[3]
            print("gt params: ", params)
        else:
            print("Must specify dummyInpFile trainSetDir and config.json:  gt inpFile trainSetDir config.json")
            exit(1)

    elif command in ['GanCompare', 'gc']:
        if len(sys.argv)==6:
            with open(sys.argv[5], "r") as f: 
                params=json.load(f)
            params['trainSetDir'] = sys.argv[3]
            params['testSetDir'] = sys.argv[4]
            print("gc params: ", params)
        else:
            print("Must specify dummyInpFile trainSetDir, testSetDir and config.json:  gc inpFile trainSetDir testSetDir config.json")
            exit(1)

    elif command in ['scanSetForSequence', 'ssfs']:
        if len(sys.argv)==5:
            params = {}
            params['inputDirectory'] = sys.argv[3]
            params['outputDirectory'] = sys.argv[4]
            print("ssfs params: ", params)
        else:
            print("Must specify inpfile, inputDirectory and outputDirectory:  ssfs inpfile inputDirectory outputDirectory")
            exit(1)

    elif command in ['surfaceAnalysisComplete','sac']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("sac params: ", params)
        else:
            print("Must specify inpfile and configfile:  sac inpfile config.json")
            exit(1)


    elif command in ['surfaceFindTube','sft']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("sft params: ", params)
        else:
            print("Must specify inpfile and configfile:  sft inpfile config.json")
            exit(1)

    elif command in ['surfaceFindGB','sfg']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("sfg params: ", params)
        else:
            print("Must specify inpfile and configfile:  sfg inpfile config.json")
            exit(1)

    elif command in ['surfacesFind','sfs']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("sf params: ", params)
        else:
            print("Must specify inpfile and configfile:  sf inpfile config.json")
            exit(1)
            
    elif command in ['solvExposure', 'solE', 'sole']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("sf params: ", params)
        else:
            print("Solve Exposure: Must specify inpfile and configfile:  sole inpfile config.json")
            exit(1)

    elif command in ['trajectoryCluster', 'tc']:
        if len(sys.argv)==5:
            with open(sys.argv[4], "r") as f: 
                params=json.load(f)
                params['inputDirectory'] = sys.argv[3]
            print("tc params: ", params)
        else:
            print("Must specify reffile, directory of PDBs and configfile:  tc inpfile directory config.json")
            exit(1)
          
    elif command in ['trajectoryMeasure', 'tm']:
        if len(sys.argv)==5:
            with open(sys.argv[4], "r") as f: 
                params=json.load(f)
                params['modelFile'] = sys.argv[3]
            print("tm params: ", params)
        else:
            print("Must specify inpfile, modelfile and configfile:  tm inpfile model config.json")
            exit(1)

    elif command in ['gyrationAnalysis', 'gyrA']:
        if len(sys.argv)==3 or len(sys.argv)==4:
            pass
        else:
            print("gyrA: Must specify inpfile:  gyrA inpfile")
            print("or gyrA: Must specify inpfile:  gyrA inpfile auxfile.xyz")
            exit(1)

    elif command in ['atomicpairwisedistance', 'apwd']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("apwd params: ", params)
        else:
            print("atomicpairwisedistance: Must specify default pdb inpfile and configfile:  apwd inpfile config.json")
            exit(1)

    elif command in ['pairwisedistance', 'pwd']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("pwd params: ", params)
        else:
            print("pairwisedistance: Must specify default pdb inpfile and configfile:  pwd inpfile config.json")
            exit(1)

    elif command in ['concatenatePDBS', 'cpdbs']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("concatenate pdbs params: ", params)
        else:
            print("concatenate pdbs: Must specify inpfile and configfile:  cpdbs inpfile config.json")
            exit(1)

    elif command in ['backboneAtomGroups', 'BBAG', 'bbag']:
        if len(sys.argv)==4:
            params = [sys.argv[3]]
        else:
            print("backboneAtomGroups: Specify inpfile and resflle:  BBAG inpfile angRange")
            exit(1)
            
    elif command in [ 'centrePDB', 'CPDB', 'cPDB', 'cpdb']:
        if len(sys.argv)!=3:
            print("cPDB: Must specify inpfile:  cPDB inpfile")
            exit(1)

    elif command in ['createSequence', 'CSQ', 'csq']:
        if len(sys.argv)!=5:
            print("csq: usage: inpfile type term(0/1)")
            exit(1)
        else:
            params=sys.argv[3:]


    elif command in ['crankShaftGroups', 'crankshaftgroups', 'csg']:
        if len(sys.argv)!=8:
            print("crankShaftGroups: usage inpfile nncutoff ddcutoff rotScale includesidechains(1/0) sidechains(1/0)")
            exit(1)
        else:
            params=sys.argv[3:]
            
    elif command in ['eliminateCIS', 'eCIS']:
        if len(sys.argv)!=3:
            print("eCIS: Must specify inpfile:   eCIS inpfile")
            exit(1)      

    elif command in ['generateCTAtomGroups', 'GCTAG']:
        if len(sys.argv)!=3:
            print("GCTAG: Must specify inpfile:   GCTAG inpfile")
            exit(1)      

    elif command in ['generateFreezeDryGroups', 'gfg', 'GFG']:
        if len(sys.argv)!=7:
            print("GCG: Must specify: GFG inpfile resIdRBody1 resIdRBody2 angRange sideChains")
            exit(1)
        else:
            params=sys.argv[3:]
            print("generateFreezeDry params: ", params)      

    elif command in ['createCTFile', 'cCTF']:
        if len(sys.argv)==5:
            params=sys.argv[3:]
            print("createCTFile params: ", params)
        else:
            print("createCTFile: Must specify inpfile, CTFile and threshold:   cCTF inpfile initCTFile thresh")
            exit(1)                  

    elif command in ['flipCT', 'fct']:
        if len(sys.argv)!=5:
            print("flipCT: Must specify inpfile, cistrans file and a force field:  fct inpfile cistran forcefield")
            exit(1)
        params = sys.argv[3:]         
    
    elif command in ['convertseq', 'ConvertSeq', 'cs', 'cS', 'CS', 'Cs']:
        if len(sys.argv)!=4:
            print("convertseq: Must specify inpfile and seq flle:  cs inpfile seqfile")
            exit(1)
        params = sys.argv[3]         

    elif command in ['multimerRg', 'mrg']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("mrg params: ", params)
        else:
            print("multimerRg: Must specify default pdb inpfile and configfile:  mrg inpfile config.json")
            exit(1)
        
    elif command in ['spherowrap', 'sw']:
        if len(sys.argv)!=4:
            print("spherowrap: Must specify inpfile and config flle:  sw inpfile config.json")
            exit(1)
        params = sys.argv[3]      
        
    # addTermini or at inpfile N(100,00) C
    elif command in ['addTermini', 'addtermini', 'AddTermini', 'at','aT','AT']:
        if (len(sys.argv)<3) or (len(sys.argv)>5):
            print("addTermini: Must specify inpfile and up to two termini (angles are optional). format N=a,b C=a,b ")
            exit(1)
        if len(sys.argv)==4:            
            params = [ sys.argv[3] ]
        if len(sys.argv)==5:
            params = sys.argv[3:]
        print("add termini params: ", params)
    
    # READ PUCKER
    elif command in ['readpucker', 'rp']:
        if len(sys.argv)==3:
            params = pl.fileRootFromInfile(infile)
            params = params+'.pucker'
        else:
            params=sys.argv[3]
        print("rp params: ", params)

    # check pucker pattern
    elif command in ['checkPuckerPattern', 'cpp', 'CPP']:
        print('No Params for check Pucker Pattern')
        # no params allowed for this command

    # Read Chirality
    elif command in ['readchirality', 'rc', 'RC','rC','Rc']:
        if len(sys.argv)==3:
            params = pl.fileRootFromInfile(infile)
            params = params + '.chi'
        else:
            params=sys.argv[3]
        print("rc params: " + params)

    # group atoms
    elif command in ['groupAtomsXYZ', 'gAX', 'gax', 'GAX']:
        if len(sys.argv)==3:
            params = pl.fileRootFromInfile(infile)
            params = params + '.xyz'
        else:
            params = sys.argv[3]
        print("gax params: " + params)

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
            print(usage)
            sys.exit()
        print("mXB params: ", params)

    # Fix Chirality
    elif command in ['fixchirality', 'fc', 'FC','fC','Fc']:
        if len(sys.argv)==3:
            params = pl.fileRootFromInfile(infile)
            params = params + '_chi.pdb'
        else:
            params = sys.argv[3]
        print("fc params: " + params)

    # ramachandran 
    elif command in ['ramachandran', 'rmc', 'rama']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("rama params: ", params)            
        else:
            print(" usage:  pdbproc rama <infile.pdb> <figConfig.json>")

    # ramachandrans 
    elif command in ['ramachandrans', 'rmcs', 'ramas']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f:
                params = json.load( f )
            print("ramachandran plots", sys.argv[3] , params)
        else:
            print(" usage:  pdbproc ramas <dummy.pdb> <figConfig.json>")

    #ramachandran density plot
    elif command in ['ramadensity', 'rmd', 'ramad']:
        if len(sys.argv)==4:
            with open(sys.argv[3], "r") as f: 
                params=json.load(f)
            print("ramad params: ", params)
        else:
            print("Must specify inpfile and configfile:  ramad inpfile config.json")
            exit(1)


    #replace pdb with xyz data
    elif command in ['replacePdbXyz','xyz2pdb']:
        if len(sys.argv)<4:
            print("ReplacePdbXyz usage: pdbproc.py xyz2pdb inpfile xyzfile [outfile]")
            sys.exit(0)
        elif (len(sys.argv)==4):
            params=[sys.argv[3]]
            params.append(pl.fileRootFromInfile(infile))
        elif (len(sys.argv)>4):
            params=sys.argv[3:5]

        print("xyz2pdb command: ")
        print(params)

    elif command in ['dumpCAsAsPDB', 'dCA']:
        print(" No params: dumping CAs as PDB: pdbproc dCA inpfile")

    #Rotate groups
    elif command in ['rotateGroup', 'RG', 'rg']:
        if len(sys.argv)<4:
            print("Rotate group usage: pdbproc.py rotateGroup inpfile atomgroups [outfile]")
            sys.exit(0)
        elif (len(sys.argv)==4):
            params = [ sys.argv[3] ]
            params.append( pl.fileRootFromInfile(infile) + '.rot.pdb' )
        elif (len(sys.argv)>4):
            params = sys.argv[3:5]

        print("rotateGroup command: ")
        print(params)

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

                    
        print(["read symmetry params: "]) 
        print(params)

    ##Pucker State Summary; has no command parameter processing; just has an infile
    elif command in ['puckerBreakDown', 'PBD', 'pbs']:
        print("pucker break down invoked")
 
    #### Convert Proline to HydroxyProline
    elif command in ['proToHyp','PH','pH','Ph','ph']:
        if len(sys.argv)<4:
            print("proToHyp: you must specify an input pdb file and a file with a list of residue to convert, and an optional output file")
            print(usage)
            sys.exit(1)
        elif len(sys.argv)==5:
            params = sys.argv[2:]
        else:
            params.append( sys.argv[3] )
            params.append( pl.fileRootFromInfile(infile) + '_hyp.pdb' )


    #### read Sequence
    elif command in ['readSequence','rsq','RSQ']:
        if len(sys.argv)<4:
            print("readSequence input.pdb mode [width]")
            exit(1)
        elif len(sys.argv)==4: 
            params=[sys.argv[3]]
        else:
            params=sys.argv[3:]
            
        params.append( pl.fileRootFromInfile(infile) + '.seq' )
        l = "readSequence params: "
        for p in params:
            l = l + str(p) + ' '
        print(l)

    #### Replace Sequence
    elif command in ['modifySequence','MS','mS','Ms','ms']:
        if len(sys.argv)<5:
            print("modifySequence input.pdb newSeq resNumStart [output.pdb]")
            print(usage)
            exit(1)
        elif len(sys.argv)==6:
            params=sys.argv[3:]
        else:
            params = [ sys.argv[3:] ]
            params.append( pl.fileRootFromInfile(infile) + '_ms.pdb' )

    #### RENAME TERMINI
    elif command in ['renameTermini','rt','rT','RT']:
        if len(sys.argv)<4:
            print("renameTermini: you must specify an input pdb file, and a flag")
            print(usage)
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
        print('renumberRes', params)

    #### removeDuplicates
    elif command in ['removeDup','rd','RD','rD']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append(sys.argv[3])
        else:
            params.append( pl.fileRootFromInfile(infile) + '_noDup.pdb' )
        print('removeDup', params)

    #### Read Residue Symmetry
    elif command in ['readResidueSymmetry','rrs','RRS']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append(sys.argv[3])
        else:
            params.append( pl.fileRootFromInfile(infile) + '.resSym' )
        print('removeDup', params)

    #### checkTorsion
    elif command in ['checkTorsion','ct','cT','Ct','CT']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append( sys.argv[3] )
        else:
            params.append( pl.fileRootFromInfile(infile) + '.torsion' )
        print('checkTorsion', params)

    #### residueInfo    
    elif command in ['residueInfo','RI','Ri','rI','ri']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append( sys.argv[3] )
        else:
            params.append( pl.fileRootFromInfile(infile) + '.residue' )
        print('residueInfo', params)


    #### torsionDiff
    elif command in ['torsionDiff','td','tD','Td','TD']:
        #check for an outputfile
        if len(sys.argv)==4:
            params.append(sys.argv[3])
        else:
            params.append('diff.torsion')
        print('torsionDiff', params)

    #### puckerGroupsRigidBody
    elif command in ['puckerGroupsRB','pgrb']:
        #no output file or flags to specify.
        params.append('rbodyconfig')

    #### puckerGroupsRigidBodys
    elif command in ['puckerGroupsRBS','pgrbs']:
        #check for an outputfile
        if len(sys.argv)<3:
            print("Usage: puckerGroupsRBS <inpfile.pdb> <resnumbersFile>")
            print(usage)
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
            print("Usage: puckergroupSpec <inpfile.pdb> <resnumbersFile> <OHFlag> <scaleFac> <probRot> [<outputfile>]")
            print(usage)
            exit(1)
        else:
            params=sys.argv[3:]
            if len(sys.argv)==7:
                params.append('atomgroups')

    #### PREP FOR AMBERGMIN
    elif command in ['prepAmberGmin','pag','PAG','pAG']:
        if len(sys.argv)<5:
            print("prepAmberGMIN: you must specify an input file, a rule file and a forcefield.")
            print(usage)
            exit(1)
        else:
            params=sys.argv[3:]

    #### SORT RESIDUES
    elif command in ['sortRes', 'sr', 'sR' ,'SR']:
        if len(sys.argv)<4:
            print("sortRes: you must specify an input PDB file and a sortfile.")
            print(usage)
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
                    print("Unrecognised input filename:" + params[0] + " or " + params[0] + ".sort")
                    exit(1)

        if len(sys.argv)>3:
            params.append( pl.fileRootFromInfile(infile) + '_sort.pdb' )
        else:
            params=sys.argv[4]

        print("sortResidue params: ")
        print(params)

    #### REMOVE ATOMS    
    elif command in ['removeatoms', 'ra']:
        if len(sys.argv)<4:
            print("removeatoms: you must specify a rule file and an input PDB file, and optionally an output file")
            print(usage)
            exit(1)
        
        if len(sys.argv)==4:
            params.append(sys.argv[3])
            params.append(sys.argv[3][0:-4]+'_ra.pdb')
            params.append(0)
           
        if len(sys.argv)==5:
            params.append(sys.argv[3])
            params.append(sys.argv[4])
            params.append(0)
        
        if len(sys.argv)==6:
            params.append(sys.argv[3])
            params.append(sys.argv[4])
            params.append(int(sys.argv[5]))
            
        
        print("ra params: ")
        print(params)

    #### WRITE PUCKER
    elif command in ['writepucker', 'wp']:
        if len(sys.argv)<4:
            print("Must specify a pucker state file and an input PDB file.")
            print(usage)
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
                print("Unrecognised input file name: " + params[0] + " or " + params[0] + ".pucker")
                exit(1)
                
        #optional outputfilename
        if len(sys.argv)==4:
            params.append( pl.fileRootFromInfile(infile)+'.pucker.pdb' )
        else:
            params.append(sys.argv[4])

        print("wp params: ")
        print(params)
    else:
        print("Unrecognised Command")
        exit(1)
        
    return [infile, command, params]

if __name__ == '__main__':

        [infile, command, params] = commandLineProc()

        if command in ['readpucker','rp']:
            pl.readpucker(infile, params)

        elif command in ['alignPdbs', 'apdb']:
            pl.alignPDBs( params )
        
        elif command in ['correlatePdbs', 'corrpdb']:
            pl.correlatePDBs( params )

        elif command in ['dumpParticularAtoms','dpa']:
            pl.dumpParticularAtoms(infile, params)

        elif command in ['pairwisedistance', 'pwd']:
            pl.pairwisedistance(infile, params)

        elif command in ['atomicpairwisedistance', 'apwd']:
            pl.pairwiseAtomicDistances(infile, params)
        
        elif command in ['concatenatePDBS', 'cpdbs']:
            pl.concatenatePdbs(infile, params)

        elif command in ['backboneAtomGroups', 'BBAG', 'bbag']:
            pl.backboneAtomGroups(infile, float(params[0]) )

        elif command in ['crankShaftGroups', 'crankshaftgroups', 'csg']:
            print(params)
            pl.crankShaftGroups(infile, int(params[0]), float(params[1]), rotScale=float(params[2]), includeSideChains=bool(int(params[3])), rotateSideGroups=bool(int(params[4])) )

        elif command in ['fragmentPDB', 'fragmentpdb', 'fpdb', 'fPDB', 'FPDB']:
            pl.fragmentPDB(infile, params)
            
        elif command in ['frenetFrameAnalysis', 'ffa']:
            pl.frenetFrameAnalysis(infile, params)
        
        elif command in ['createSequence', 'CSQ', 'csq']:
            pl.createSequence(infile, type=params[2], term=params[1])
        
        elif command in [ 'centrePDB', 'CPDB', 'cPDB', 'cpdb']:
            pl.centrePDB2(infile)
        
        elif command in ['generateCTAtomGroups', 'GCTAG']:
            pl.generateCTAtomGroups(infile)

        elif command in ['generateFreezeDryGroups', 'gfg', 'GFG']: 
            pl.generateFreezeDryGroups(infile, int(params[0]), int(params[1]), float(params[2]), bool(params[3]) )

        elif command in ['gyrationAnalysis', 'gyrA']:
            print(sys.argv)
            try:
                if sys.argv[3]:
                    print(f"two file route: {infile}, {sys.argv[3]}")
                    pl.gyrationAnalysis(infile, auxiliaryFile=sys.argv[3])
            except IndexError:
                print(f"one file route: {infile}")
                pl.gyrationAnalysis(infile)
        
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
        
        elif command in ['multimerRg', 'mrg']:
            pl.multimerRg(infile, params)
        
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

        elif command in ['GanTrain', 'gt']:
            pl.GanTrain(**params)

        elif command in ['GanCompare', 'gc']:
            pl.GanCompare(**params)        

        elif command in ['measureTubeRadius','mtr']:
            pl.measureTubeRadius(infile, params)

        elif command in ['ramachandran', 'rmc', 'rama']:
            pl.ramachandran(infile, params)

        elif command in ['ramadensity', 'rmd', 'ramad']:
            pl.ramaDensity(infile, params)
        
        elif command in ['ramachandrans', 'rmcs', 'ramas']:
            pl.AllFilesRamachandran( params )
        
        elif command in ['renameTermini','rt','rT','RT']:
            pl.renameTerminiTop(params[1],infile, int(params[0]))
        
        elif command in ['renumberRes','rn','rN','RN']:
            pl.renumberResidues(infile,params[0],params[1])

        elif command in ['scanSetForSequence', 'ssfs']:
            pl.scanSetForSequence(infile, params['inputDirectory'], params['outputDirectory'])
        
        elif command in ['sortRes','sr','sR','SR']:
            pl.sortResidues(infile, params[0], params[1])
        
        elif command in ['spherowrap', 'sw']:
            pl.spherowrap(infile, params)

        elif command in ['trajectoryCluster', 'tc']:
            pl.trajectoryCluster(infile, params)
          
        elif command in ['trajectoryMeasure', 'tm']:
            pl.trajectoryMeasure(infile, params)
        
        elif command in ['dumpCAsAsPDB', 'dCA']:
            pl.dumpCAsAsPDB(infile)
        
        elif command in ['surfaceAnalysisComplete','sac']:
            pl.surfaceAnalysisCompleteGlob(infile, params)

        elif command in ['surfaceFindGB','sfg']:
            pl.surfaceFindGB(infile, params)

        elif command in ['surfaceFindTube','sft']:
            pl.surfaceFindTube(infile, params)
        
        elif command in ['surfacesFind','sfs']:
            pl.surfacesFind(infile, params)

        elif command in ['solvExposure', 'solE', 'sole']:
            pl.solventExposure(infile, params)
            
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
            print("Unknown Command: " + command)

        print("Done")