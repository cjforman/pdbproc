from pele.playground.molecule import MolecularSystem
import numpy as np
import os, ftplib, gzip
import re

class ProteinSystem(MolecularSystem):
    """
    Defines a representation of a protein and provides a set of operations that can be performed
    on the representations.  Extends the base Molecule class, which extends the BaseSystem Class.
    
    Acceptable input to define the protein is: 
    a PDB filename
    a PDB code
    A sequence as a list of three letter codes or a string of single letter codes. 
        (Generates an unfolded structure using a standard library of residues)  
    
    The representation of the protein is a set of five related lists: 
    list of atomic objects, list of bond objects and list of interesting groups as defined by the base molecular class.         
    In addition we add
    
    list of residue objects      id number, sequence index, a list of atoms, residue name  
    list of residue chains       id number, chain Name, list of residue objects in chain. 
    
    Allowed operations in addition to those defined in the base molecular class are:
     
    get back bone dihedrals:      returns a list of back bone dihedral angles for each residue.
    
    get back bone dihedrals:      sets back bone dihedral angles for specified residue.
    
    mutate side chains:           mutate a specific residue - retains the C, N, O, CA and CB atoms, and asserts 
                                  a new side chain as defined from the internal library of side chains.
                                  
    re-thread protein:            provide a new sequence for all or part of the protein while retaining 
                                  the old backbone shape.
                                  
    renumber residues:            renumber the external residue numbers starting at a given offset. 
    
    renders graphics              additional functions to draw higher order representations of the protein using the open GL viewer.
    
    save a PDB                    saves a standard PDB file with enough information to reproduce the representation. 
    
    output a sequence             saves the sequence to file to allow easy modifications and re-engineering
    
    download a PDB file           given a PDB code, download a specific PDB from the protein database. 
    
    add fragments                 combines two representations using protein level information (i.e. residue numbers of chains)
    
    
    Log
    ===
    4th July 2014    Created by CJF41
    
    """
    def __init__(self,inputString=None):
        '''
        Function determines if the input string is a PDB code, a PDB file, a sequence of residues or a list of 3 or 1 letter
        residue codes. Depending on the result a protein representation is initialized in the most appropriate way. 
        If it is simply sequence information then an internal library of atomic coords for each residue is used to 
        construct an unfolded protein. If the generation of the representation fails then the 
        self.validRepresentation flag is set to False.
        '''
        
        #set up various things  
        self.createAminoConversionCode()
        
        #Determine the input type and act accordingly.
        self.determineInputType(inputString)
        if self.mode=='PDBFILE':
            self.createRepresentationFromPDBData()
        if self.mode=='SEQUENCE':
            self.createRepresentationFromSequence()
        if self.mode=='UNKNOWN':
            self.validRepresentation=False



    def createRepresentationFromPDBData(self):
        '''
        Parse a PDB raw Data file, create a list of atom objects, bond objects, residue objects and chain objects
        from a PDB file and then create a representation object to store it all neatly within the parent protein 
        system object. 
        '''
        
        #check that the raw_data has been loaded
        if self.rawData:
            atoms=[self.parsePdbLine(self,line) for line in self.rawData if line.split()[0] in ['ATOM','HETATM']]
            
            #create a list of atom objects 
            atomObjects=[Atom(index,atom[0],atom[1],atom[8],atom[5],atom[6],atom[7]) for (index,atom) in enumerate(atoms)]
            
            #create a set of unique chain Ids from the PDB
            chainInfo=set([ atom[3] for atom in atoms])
            
            #create a set of unique (chain Id, residue number, residue name) triples from the PDB
            resInfo=set([ (atom[3],atom[4],atom[2]) for atom in atoms])
            
            
                    atomNum,atomName,ResidueName,ChainId,residueNumber,atomCoordsX,atomCoordsY,atomCoordsZ,atomSymbol
                    atomID, molIndex, atomName, atomType, x, y, z
            
            
        else:
            self.validRepresentation=False
        
        
        
    def pdbLineFromAtom(self):
        try:
            l='ATOM {: >06d} {: <4}{:1}{:3} {:1}{: >4d}{:1}   {: >8.3f}{: >8.3f}{: >8.3f}{: >6.2f}{: >6.2f}      {: <4}{: >2}{: >2}\n'.format(self.atomNum,self.atomName,self.AtlLoc,self.ResidueName,self.ChainId,self.ResidueNumber,self.ResInsertCode,self.atomCoordsNP[0],self.atomCoordsNP[1],self.atomCoordsNP[2],self.occupancy,self.tempFactor,self.segId, self.atomSymbol,self.charge)
        except:
            l="unable to write atom to string"
        return l
        
    
    def parsePdbLine(self,line):
        #Read each desired parameter from a pdb line
        atomNum = int(line[6:11])
        atomName = line[12:16].split()[0]
        ResidueName = line[17:20].split()[0]
        ChainId = line[21]
        residueNumber = int(line[22:26])
        atomCoordsX = float(line[30:38])
        atomCoordsY = float(line[38:46])
        atomCoordsZ = float(line[46:54])
        try:
            atomSymbol = line[76:78].split()[0]
        except:
            atomSymbol =next((symbol_ for symbol_, name_ in self.atomicSymbols.items() if name_==atomName),None) 
            
        return (atomNum,atomName,ResidueName,ChainId,residueNumber,atomCoordsX,atomCoordsY,atomCoordsZ,atomSymbol)
        
    def determineInputType(self,inputString):
        '''
        Determines if the input string is a sequence of residues, a single file name or a PDB code.
        Sets self.mode as either PDBFILE, SEQUENCE or UNKNOWN 
        If it is a PDB file the data loads it into a member variable called self.rawData.  
        If it is a PDB code, the data is downloaded and loaded into self.rawData 
        If it is a sequence the sequence is stored as self.sequenceOne and self.sequenceThree
        as both one and three letter codes. Probably over kill but what the heck. The conversion
        acts as a check to ensure the sequence is sensible, since we computed both versions we might as well keep it.
        '''
        #Assume we will fail.
        self.mode='UNKNOWN'
        
        #Check to see if the input string is a single unicode base string
        if isinstance(inputString, basestring):
            
            #compiles regexp for a valid PDB code (4 chars a single numeral followed by 3 alphanumerics)
            regExpPDBCode=re.compile('[0-9][a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]')
            
            #if input string is 4 chars long and matches the refExp it is a valid PDB code.
            if len(inputString)==4 and not regExpPDBCode.match(inputString)=='None': 
                
                #attempt to download pdb file from PDB. (if this failed then the attempt to load the file will also fail)
                self.pdbDownload(inputString)
                
                #attempt to load downloaded file
                self.rawData=self.readTextFile(self,self.pdbFilename)
                if self.rawData:
                    self.mode = 'PDBFILE'              
                else:
                    self.rawData=self.readTextFile(self,inputString)
                    #attempt to load <inputString> (could have been just a four letter filename rather than a PDB code for download!))
                    if self.rawData:
                        self.mode = 'PDBFILE'
                        self.pdbFilename=inputString
            else:
                #Not a PDB code so check to see if it is a valid SEQUENCE of amino acid residues.
                regExpPDBCode=re.compile('^[ROHKDESTNQCUGPAVILMFYW]+$')
                if regExpPDBCode.match(inputString):
                    #But it could have been intended as a filename so attempt to load <inputString>
                    self.rawData=self.readTextFile(self,inputString)
                    
                    #if the load failed then we definitely have a SEQUENCE. If the load succeeds it was a filename!
                    if self.rawData:    
                            self.mode='PDBFILE'
                            self.pdbFilename=inputString                            
                    else:
                            #convert the sequence from a one to a three code sequence. If this fails sequenceThree
                            #is set to False
                            self.convertAminoCode(inputString)
                            if self.sequenceThree:
                                self.mode='SEQUENCE'
                else:
                    #input string is not a valid PDB code or a sequence of residues, so inputString must be a filename 
                    #or gibberish.
                    self.rawData=self.readTextFile(self,inputString)
                    if self.rawData:
                        self.mode='PDBFILE'
                        self.pdbFilename=inputString                        
                    else:
                        #last attempt before giving up! Attempt to load <inputstring>.pdb 
                        self.rawData=self.readTextFile(self,inputString+'.pdb')
                        if self.rawData:
                            self.mode='PDBFILE'
                            self.pdbFilename=inputString+'.pdb'                            
        else:
            #check to see if the input string is a list
            if (isinstance(inputString,list)):
                #attempt to convert the list into a sequence of one letter amino acid codes,
                self.convertAminoCode(inputString)
                if self.sequenceOne:
                    self.mode='SEQUENCE'
    
    def unZip(self,some_file,some_output):
        """
        Unzip some_file using the gzip library and write to some_output.
        """
         
        f = gzip.open(some_file,'r')
        g = open(some_output,'w')
        g.writelines(f.readlines())
        f.close()
        g.close()
    
        os.remove(some_file)

    def pdbDownload(self,pdbCodes):
        """
        Download all pdb files in pdbCodes and unzip them. Based on code written by 
        Michael J. Harms and distributed under General Public License v. 3.
        Copyright 2007.
        
        Grows a list of downloaded and unzipped filenames. If the file failed to download 
        then the entry corresponding to that filename is set to False. 
        """
        
        hostname="ftp.wwpdb.org"
        directory="/pub/pdb/data/structures/all/pdb/"
        prefix="pdb"
        suffix=".ent.gz"
    
        # Log into server
        print "Connecting..."
        ftp = ftplib.FTP()
        ftp.connect(hostname)
        ftp.login()
        
        #set up output
        self.filename=[]
        
        # Download all files in file_list
        to_get = ["%s/%s%s%s" % (directory,prefix,f,suffix) for f in pdbCodes]
        to_write = ["%s%s" % (f,suffix) for f in pdbCodes]
        for i in range(len(to_get)):
            try:
                ftp.retrbinary("RETR %s" % to_get[i],open(to_write[i],"wb").write)
                final_name = "%s.pdb" % to_write[i][:to_write[i].index(".")]
                self.unZip(to_write[i],final_name) 
                self.filename.append(final_name)
            except ftplib.error_perm:
                os.remove(to_write[i])
                self.filename.append(False)
    
        # Log out
        ftp.quit()
    
    def createAtomicSymbolCode(self):
        'generate the dictionary for relating atomic names to atomic types.'
        self.standardSymbols={ 'C':['CA','CB','CD','CE','CF','CG'],
                                    'H':['H1','H2','H3','OH1','OH2','1H','2H','3H','HA','HB1','HB2','HB3'],
                                    'O':'O'}

    
    def createAminoConversionCode(self):
        'generate the dictionary for converting between amino acid codes'
        self.standardResidueCodes={ 'GLY':'G',
                                    'PRO':'P',
                                    'HYP':'O',
                                    'ALA':'A',
                                    'VAL':'V',
                                    'LEU':'L',
                                    'ILE':'I',
                                    'MET':'M',
                                    'CYS':'C',
                                    'PHE':'F',
                                    'TYR':'Y',
                                    'TRP':'W',
                                    'HIS':'H',
                                    'LYS':'K',
                                    'ARG':'R',
                                    'GLN':'Q',
                                    'ASN':'N',
                                    'GLU':'E',
                                    'ASP':'D',
                                    'SER':'S',
                                    'THR':'T'}
    
    def convertAminoCode(self,sequence):
        '''
        Function converts the given sequence from a one letter code to a three letter code or vice versa
        depending on which one is given.
        Both codes are written to self.sequenceOne and self.SequenceThree respectively. 
        If the process fails then sequenceOne and sequenceThree are set to false.
        '''
        #initialise output
        self.sequenceOne=''
        self.sequenceThree=[]
        
        #if we given a list, assume it is a list of three letter codes.
        if isinstance(sequence,list):
            #loop through the sequence
            for three in sequence:
                #get the one letter code from the dictionary. returns None if three is invalid
                one=self.standardResidueCodes.get(three)
                if not one == None:
                    self.sequenceOne+=one
                    self.sequencethree.append(three)
                else:
                    self.sequenceOne=False
                    self.sequenceThree=False
                    break
        
        #if sequence is a string then assume it is a string of one letter codes.
        if isinstance(sequence,basestring):
            #loop through the sequence
            for one in sequence:
                #get the three letter code from the dictionary (uses it in reverse!). returns None if one is not there
                three=next((three_ for three_, one_ in self.standardResidueCodes.items() if one_==one),None)
                if not three == None:
                    self.sequenceOne+=one
                    self.sequencethree+=three
                else:
                    self.sequenceOne=False
                    self.sequenceThree=False
                    break
    
    def readTextFile(self,filename):
        #read line data in from text file into the rawData member variable
        try:
            vst = open(filename, 'r')
            self.rawData=vst.readlines()
            vst.close()
        except IOError as e:
            self.rawData=False

    
class Atom(object):
    """
    A minimal representation of an atom:
        imolecular Index, name, type, atom X, atom Y, atom Z
         
    """
          
    def __init__(self, atomID, molIndex, atomName, atomType, x, y, z):
        self.atomID=atomID
        self.molIndex=molIndex
        self.atomName=atomName
        self.atomType=atomType
        self.coordNP=np.array([x,y,z])
        self.coordXYZ=[x,y,z]
        
class bond(object):
    ''' 
    A single bond within a molecule. Minimal definition of a bond. atom1 and atom2 are two atom objects.
    '''
    def __init(self,idNumber,atom1,atom2):
        self.idNumber=idNumber
        self.atom1=atom1
        self.atom2=atom2


class Residue(object):
        ''' 
        A single residue within a protein structure.
        
        Has a name, a resid and a list of atom objects associated with it
        '''
        def __init__(self,resId, resName, atomList): 
            self.resId=resId
            self.resName=resName
            self.atomList=atomList
            self.numAtoms=len(atomList)
    
class Chain(object):
        ''' 
        A single chain within a protein structure.
        
        Has an identifier and a list of residue objects associated with it
        '''
        def __init__(self,chainId, resList): 
            self.chainId=chainId
            self.resList=resList
            self.numResidues=len(resList)
    

class ProteinRepresentation(object):
        '''
        A class that forms the basic internal representation of a protein 
        or protein fragment used throughout the system.
        '''
        def __init__(self,listChains,listResidues,listBonds,listAtoms):
            
            self.listChains=listChains
            self.listResidues=listResidues
            self.listBonds=listBonds
            self.listAtoms=listAtoms
        
