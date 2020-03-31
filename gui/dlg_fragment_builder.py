import pickle

import copy as cp
import numpy as np
from PyQt4 import QtGui, QtCore, Qt

import pdbLib as pdb 

from  ui_fragment_builder import Ui_MainWindow
from dlg_fragParams import DlgFragParams


class FragmentItem(QtGui.QListWidgetItem):    
    """A class for storing information pertaining to a fragment"""
    def __init__(self, params,fragId):
        #initialise member variables
        self.name=params.get("Fragment_Name")
        self.filename=params.get("Pdb_Filename")
        self.firstRes=params.get("Pdb_First_Residue")
        self.lastRes=params.get("Pdb_Last_Residue")
        self.NConnector=params.get("Staple_Residue_N")
        self.CConnector=params.get("Staple_Residue_C")
        self.fragId=fragId

        #load the atomic data from the pdb file into the fragment.
        self.populateAtoms()
        self.Compute_COG()
        
        QtGui.QListWidgetItem.__init__(self, str(self.fragId)+' '+self.name)
 
    def get_CoordsNP(self):
        return self.CoordsNP
 
    def get_CoordsXYZ(self):
        return self.CoordsXYZ
 
    def get_COG(self):
        return self.COG
 
    def populateAtoms(self):
        """
        loads the atoms data from a pdb file into the fragment object and performs basic validation on the fragment
        """
        #reads atoms using the pdb processing library.
        atoms = pdb.readAtoms(self.filename)
        
        #extracts the atoms from the pdb data that are between the specified residues
        self.atoms=[ atom for atom in atoms if atom[5]>=self.firstRes and atom[5]<=self.lastRes]
        
        #store the atomic position data in two different ways - one is the standard way for GMIN the other a numpy array        
        self.CoordsNP= [ np.array([atom[7],atom[8],atom[9]]) for atom in self.atoms]
        self.CoordsXYZ = []
        for atom in self.atoms:
            self.CoordsXYZ.append(atom[7])
            self.CoordsXYZ.append(atom[8])
            self.CoordsXYZ.append(atom[9])
    
    def Compute_COG(self):
        """
        Computes the centre of gravity of the Fragment
        """
        if len(self.CoordsNP)>0:
            self.COG=sum(self.CoordsNP)/len(self.CoordsNP)


class FragmentBuilder(QtGui.QMainWindow):
    """
    the GUI for exploring normal modes
    """
    def __init__(self, parent=None, system=None, app=None):
        QtGui.QMainWindow.__init__(self, parent=parent)
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.view3D.setSystem(system)
        self.system = system
        
        self.app = app
        self.current_selection = None
        self.numFragments=0
        self.maxFragId=0
        self.numAtoms=0
        
    def on_listFragments_currentItemChanged(self, newsel):
        """
        change which fragment we're looking at
        """
        if newsel is None:
            self.currentmode = None
            return
        self.current_selection = newsel
        self.coords=cp.copy(newsel.get_CoordsXYZ())
        self.ui.view3D.setCoords(self.coords,1)

    def on_actionAdd_Fragment_triggered(self, checked=None):
        """
        open a modal dialog box to get the parameters that define a fragment
        """
        if checked is None:
            return
        #set up a dictionary to hold the fragment items.
        fragParams=dict()
        fragParams["Fragment_Name"]= 'L1'
        fragParams["Pdb_Filename"] = '2RNM_small.pdb'
        fragParams["Pdb_First_Residue"]=218
        fragParams["Pdb_Last_Residue"]=226
        fragParams["Staple_Residue_N"]=218
        fragParams["Staple_Residue_C"]=226
        
        #create a modal dialog box which displays the dictionary and allows the user to modify the values.
        paramsdlg = DlgFragParams(fragParams, parent=self)
        result=paramsdlg.exec_()

        #if the code in the dialog box was accepted then create the fragment object using the params dictionary as input        
        if (result):
            #code that creates a fragment item from the dictionary, adds it to the Fragment List and sets focus to that item
            self.ui.listFragments.addItem(FragmentItem(fragParams,self.maxFragId+1))
            self.numFragments+=1
            self.maxFragId+=1
            
    def on_actionRemove_Fragment_triggered(self, checked=None):
        """
        open a modal dialog box to get the parameters that define a fragment
        """
        if checked is None:
            return

        #remove any selected fragments.
        for SelectedItem in self.ui.listFragments.selectedItems():
            self.ui.listFragments.takeItem(self.ui.listFragments.row(SelectedItem))
            
                    
if __name__ == "__main__":
    from OpenGL.GLUT import glutInit
    import sys
    glutInit()
    from pele.systems import LJCluster
    system = LJCluster(13)
    app = QtGui.QApplication(sys.argv)
    
    wnd = FragmentBuilder(app=app,system=system)
    
    wnd.show()
    sys.exit(app.exec_())     
