import sys
from Utilities.keyProc import keyProc
import numpy as np
import Utilities.fileIO as fIO
import Utilities.cartesian as cart
import random as rnd
import Builder.BuildingBlock as BuildingBlock

class molecularStapler(keyProc):
    # A class to concatenate a series of building blocks according to a sequence of instructions in a file.
    # There are four kinds of instructions which are BUILDINGBLOCK, STAPLE, STARTBLOCK and STAPLESEQUENCE.
    # BUILDING BLOCK consists of a filename and a series of n lots of 3 indices. 
    # Each set of three indices defines a connector, which are numbered in the order they are defined.
    # STAPLES are also building blocks and are defined in the same way but they only have two connectors. 
    # STARTBLOCK is a BUILDING Block but it only has one connector and is the terminal connector.
    # The staple sequence defines which building blocks are connected to which staples using which connectors and so on.
    # DoStaple applies the StapleSequence to the building blocks and outputs a single xyz file.
    
    def __init__(self, paramFilename):
        keyProc.__init__(self, paramFilename)
        
        # assume all will be well.
        self.allGood = True
        
        # set up any subdictionaries for the parameter structure
        self.params['BUILDINGBLOCK'] = []
        self.params['STAPLE'] = []
        self.params['STARTBLOCK'] = []        
        self.params['STAPLESEQUENCE'] = []
        
        if self.noErrors==False:
            print "Error loading parameters"
            self.dumpParams()
            self.allGood = False 
        else:
            print "Successfully initiated moelcular Builder object construction"
        
    def isAllGood(self):
        return self.allGood
    
    def initialiseParameters(self):
        self.buildingBlocks = [ BuildingBlock(tuple(buildingBlockData)) for buildingBlockData in self.getParam('BUILDINGBLOCK')] 
        self.stapleBlocks = [ BuildingBlock(tuple(stapleData)) for stapleData in self.getParam('STAPLE')]        
        self.workingBlock = BuildingBlock(self.getParam('STARTBLOCK')[0]) # should only ever have one start block
        self.sequence = [ tuple(map(int, sequence)) for sequence in self.getParam('STAPLESEQUENCE')]

    def processInstructions(self):
        for instruction in self.sequence:
            self.DoStaple(instruction)

    def exportWorkingBlock(self):
        atoms, names = self.workingBlock.getAtoms()
        fIO.saveXYZList(atoms, names, self.paramFilename[0:-4]+'.xyz')
        print "Saving file: " + self.paramFilename[0:-4] + ".xyz"
        
    def DoStaple(self, instruction):
        # create names for the instructions to make code clearer  
        # Subtract 1 because indices are zero based but the command file is 1 based. seems more user friendly that way.
        workingBlockConnector = instruction[0] - 1
        staple = instruction[1] - 1
        buildingBlock = instruction[2] -1
        buildingBlockConnector = instruction[3] - 1
        buildingBlockConnectorThatBecomesNewWorkingBlockConnector = instruction[4] - 1
                
        # connect staple connector 0 to the specified workingblock connector (first specified block is the static block)
        self.makeConnection(self.workingBlock, workingBlockConnector, self.stapleBlocks[staple], 0)

        # connect specified buildingblock connector to staple connector 1
        self.makeConnection(self.stapleBlocks[staple], 1, self.buildingBlocks[buildingBlock], buildingBlockConnector)

        # add a copy of the connected building block atoms to the workingblock atoms, and deal with the connectors 
        self.weldConnection(    self.buildingBlocks[buildingBlock], 
                                workingBlockConnector, 
                                buildingBlockConnector, 
                                buildingBlockConnectorThatBecomesNewWorkingBlockConnector)
    
    def weldConnection(self, buildingBlock, wbConnection, bbConnection, newConnector):
        ''' Copies the atom information from a building block and adds it to the working block. 
            Creates a new building block connector and removes the two connectors that were just used to make the join.'''
        
        # extract the building block raw data
        xyzVals, names = buildingBlock.getAtoms()
        
        # get the current number of atoms in the working block
        numAtomsWorkingBlockOld = self.workingBlock.countAtoms()
        
        # add the new values to the working block
        self.workingBlock.addAtoms(xyzVals, names)
    
        # compute the new working block indices of all the connectors on the building block we just added 
        buildingBlockConnectors = [ [ c + numAtomsWorkingBlockOld for c in connector] for connector in buildingBlock.getAllConnectionIndices() ]
        
        # add all the building block connectors to the working block 
        # except the one that has been designated to replace the old working block connector (newConnection)
        # specifying newConnection in the parameter file as -1 terminates the chain
        # and the one that was used to link to the working block (bbConnection)  
        for i, connector in enumerate(buildingBlockConnectors):
            if i == newConnector:
                self.workingBlock.replaceConnector(wbConnection, connector)
            elif i == bbConnection:
                pass # do nothing, effectively deleting the connection used to join the bb to the wb.
            else:
                # copy across all the unaffected connectors from the bb to the wb with their new indicies
                # give them new connector numbers in the working block scheme.
                self.workingBlock.addConnector(connector)
    
    def makeConnection(self, staticBlock,  staticConnector, mobileBlock, mobileConnector):
        ''' Use the points in each block specified by the connector to calculate
        the necessary translation and rotation to align the mobileBlock to the static Block.
        
        The first point of the mobile connector is placed on the first point of the static connector
        to form a common point.
        
        A rotation is made in the plane formed by the common point and the two second points, such that the second points
        are co-linear.
        
        Then a rotation is made about line between the first and second points until the vector 
        from the second point to the third point of the mobile block is parallel with the equivalent vector in the static block.
        
        If the connectors are idential - as they would be with a well designed staple, then this will align 
        the mobile block with the static block.
        This algorithm will always give an answer but if the staples are not well designed then the resulting position of the
        building block could be bizarre. It is intended to work with a staple which is formed of three points from one building block
        and 3 points taken from another building block.'''
        
        # randomly spin the mobile block - if the connectors start off aligned then there are issues.
        rnd.seed() # reseed the random number generator
        mobileBlock.rotateBBArbitraryAxis(mobileBlock.getCOM(), np.array([rnd.random(), rnd.random(),rnd.random()]), rnd.random() * 2 * np.pi )
                
        # First get the points of interest 
        staticXYZ = staticBlock.getConnection(staticConnector)
        mobileXYZ = mobileBlock.getConnection(mobileConnector)

        # compute translation and perform it on the mobileXYZ connector to make the first points coincide.
        transVec = staticXYZ[0] - mobileXYZ[0] 
        mobileXYZ = mobileXYZ + transVec
        
        # compute the normalised vectors between the points in the connectors.
        # U is from first to second point and 
        # V is from second to third point. 
        
        mobileU = mobileXYZ[1] - mobileXYZ[0]
        mobileUHat = mobileU/np.linalg.norm(mobileU)
        staticU = staticXYZ[1] - staticXYZ[0]
        staticUHat = staticU/np.linalg.norm(staticU)
        
        # now compute first rotation axis by taking cross product of vectors between 0th and 1st points on each connector.
        rotAxis1 = np.cross(mobileUHat, staticUHat)
        rotAxis1Hat = rotAxis1/np.linalg.norm(rotAxis1)
        
        # compute first rotation angle by taking dot product of U vectors
        rot1Angle = np.arccos(np.dot(mobileUHat, staticUHat))
        
        # now rotate the mobile connector about rotAxis1 placed at it's first point - 
        # this achieves a rotation in the plane formed by the coincident point and 
        # the two U vectors of mobile and static, such that the two U vectors become aligned.
        mobileXYZ = [ cart.rotPAboutAxisAtPoint(p, mobileXYZ[0], rotAxis1Hat, rot1Angle) for p in mobileXYZ]

        # compute the v vectors now the mobile connector has been rotated. 
        mobileV = mobileXYZ[2] - mobileXYZ[1]
        mobileVHat = mobileV/np.linalg.norm(mobileV)
        staticV = staticXYZ[2] - staticXYZ[1]
        staticVHat = staticV/np.linalg.norm(staticV)

        # compute second rotation angle by taking dot product of the V vectors when their U vectors are aligned.
        rot2Angle = np.arccos(np.dot(mobileVHat, staticVHat)) 
        
        # now we have all the transformation parameters, perform the sequence on the mobileBlock itself
        # align the first point
        mobileBlock.translateBB(transVec)
        mobileBlock.export('mobBlockTrans.xyz')
        
        # rotate mobile connector about rotAxis1 place at common point (aligns the U vectors)
        mobileBlock.rotateBBArbitraryAxis(staticXYZ[0], rotAxis1Hat, rot1Angle)
        mobileBlock.export('mobBlockTransRot1.xyz')

        # rotate mobile building block about staticUHat placed at common point (aligns the V vectors)
        mobileBlock.rotateBBArbitraryAxis(staticXYZ[0], staticUHat, -rot2Angle)
        mobileBlock.export('mobBlockTransRot2.xyz')
        
        # the mobile block should now be aligned with the static connector.

    
def testDoStaple():    
    stapler = molecularStapler("testStapler.txt")
    stapler.processInstructions()
    stapler.exportWorkingBlock()
    
def testMakeConnection():
    # create Builder object and check that the makeConnection works for the staple
    stapler = molecularStapler("testStapler.txt")
    stapler.makeConnection(stapler.workingBlock, 0, stapler.stapleBlocks[0], 0)
        
def testBuildingBlock():
    
    testBlock = BuildingBlock(('testBlock1.xyz', 1, 2, 3, 2, 3, 4))
    testBlock.translateBB(np.array([2.0, 2.0, 2.0]))
    testBlock.export('testBlockTrans.xyz')
    testBlock.reset()
    print "translation complete"
    
    testBlock.rotateBBArbitraryAxis(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0]), 45*np.pi/180.0)
    testBlock.export('testBlockRot.xyz')
    testBlock.reset()
    print "rotation1 complete"
    
    
    testBlock.rotateBBAtomAxis(0, 1, 45*np.pi/180.0)
    testBlock.export('testBlockRotAtom.xyz')
    testBlock.reset()
    print "rotation2 complete"
    
    testBlock.rotateBBArbitraryAxis(testBlock.getCOM(), np.array([0.0, 0.0, 1.0]), 45*np.pi/180.0)
    testBlock.export('testBlockRotCom.xyz')
    testBlock.reset()
    print "rotation3 complete"
        
    testBlock.placeAtom(0, np.array([0,0,0])) 
    testBlock.export('testBlockPlaced.xyz')
    testBlock.reset()
    print "placement complete"
    
    testBlock.centerAtom(3)
    testBlock.export('testBlockCenterAtom3.xyz')
    testBlock.reset()
    print "centering on atom complete"
    
    testBlock.centerBB()
    testBlock.export('testBlockCentered.xyz')
    testBlock.reset()
    print "centering complete"
    
    testBlock.centerBB()
    testBlock.rotateBBArbitraryAxis(testBlock.getCOM(), np.array([0.0, 0.0, 1.0]), 45*np.pi/180.0)
    testBlock.rotateBBArbitraryAxis(testBlock.getCOM(), np.array([1.0, 0.0, 1.0]), -76*np.pi/180.0)
    testBlock.export('testBlockStapler.xyz')
    testBlock.reset()
    
    print "testBlockForStaplerCreated"
    
    stapleBlock = BuildingBlock(('testStaple.xyz', 1, 2, 3, 2, 3, 4))
    stapleBlock.rotateBBAtomAxis(0, 1, 45)
    stapleBlock.rotateBBArbitraryAxis(stapleBlock.getCOM(), np.array([0.0, 0.0, 1.0]), 53)
    stapleBlock.export('testStaplerRot.xyz')
    
    print "testStapleForStaplerCreated"
    
    print testBlock.getCOM() 
    
if __name__=="__main__":
    
    #testBuildingBlock()
    #testMakeConnection()
    #testDoStaple()
    
    # create Builder object and do the stapling
    stapler = molecularStapler(sys.argv[1])
    
    if stapler.isAllGood():
        stapler.processInstructions()
        stapler.exportWorkingBlock()
    
    
    