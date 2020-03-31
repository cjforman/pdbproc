       PROGRAM ANALYSEPDB
       implicit none   


       DOUBLE PRECISION :: Rmat1(1:3,1:3),Rmat2(1:3,1:3), dCCOG1(1:3), dCCOG2(1:3), CenteredPos(1:3), CenteredPosRot(1:3), RMat(1:3,1:3), RMatC(1:3,1:3)
       DOUBLE PRECISION :: COG1CB(1:3), b1CBp(1:3), a1d, a1dSum, b1d, c1d, c1dSum, COG1O(1:3), b1Op(1:3)
       DOUBLE PRECISION :: COG2CB(1:3), b2CBp(1:3), a2d, a2dSum, b2d, c2d, c2dSum, COG2O(1:3), b2Op(1:3)
       DOUBLE PRECISION :: CCOG1(1:3), CCOG2(1:3), b1C(1:3), b1CA(1:3), b1CB(1:3), b1N(1:3), b1O(1:3),p1(1:3),p2(1:3)
       DOUBLE PRECISION :: b2C(1:3), b2CA(1:3), b2CB(1:3), b2N(1:3), b2O(1:3) 
       DOUBLE PRECISION :: xAxis(1:3),yAxis(1:3),zAxis(1:3),b1(1:3),b2(1:3), b2XY(1:3), a1(1:3), a2(1:3), c1(1:3), c2(1:3)
       DOUBLE PRECISION :: CAB(1:3), CABac(1:3), SPOS1(1:3), SPOS2(1:3), a1g(1:3), c1g(1:3), a2g(1:3), c2g(1:3)
       DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: gamma1, gamma2, cArray
       DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: CPOS2, CAPOS2, CBPOS2, NPOS2, OPOS2, OCOG, CCOG1Array, CCOG2Array
       DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: CPOS1, CAPOS1, CBPOS1, NPOS1, OPOS1, BBX, BBY, BBZ, BBCOG
       DOUBLE PRECISION, DIMENSION(:,:,:), ALLOCATABLE :: X, Y, Z, RMATU
       INTEGER, DIMENSION(:), ALLOCATABLE :: CBPOS1Flag, CBPOS2Flag, numArrowsPerModel

       DOUBLE PRECISION :: Occupancy, TempFact, xDum, yDum, zDum, d, beta1, beta2, alpha, gamma1Av, gamma2Av, b2DotZ
       DOUBLE PRECISION :: dHorizModAv, dVertModAv, alphaModAv, beta1ModAv, beta2ModAv, gamma1ModAv, gamma2ModAv
       DOUBLE PRECISION :: aAModAv, bAModAv, cAModAv, aRModAv, bRModAv, cRModAv
       DOUBLE PRECISION :: CABb1, CABb2, CylWidth, ConeWidth, pi,xDotb,CABaca1, CABacz1, b2XYdotx, b2XYdoty, CABaca2, CABacc2
       INTEGER ::  numArgs, nBB, curBB, nRb, nRe, nRS1, nRS2, nRS1b, nRS1e, nRS2b, nRS2e, curRS1, curRS2, atomNum, numModel, curModel
       INTEGER ::  residueNum, curResidue, nA1, nA2, curR, nB1, nB2, continueFlag, reason, UNITOK
       CHARACTER(len=256):: inpFilename, outFilename
       CHARACTER(len=256):: atomType 
       CHARACTER(len=256):: residueName, SegmentId, ElementType
       CHARACTER:: chain, insert
       CHARACTER(len=1000):: buffer
       CHARACTER(len=8):: fileNumStr

       !initialisations
       pi=3.1415926535897932
       CylWidth=0.1
       ConeWidth=0.2

      !read command line argument to get filename of PDB database file
      numArgs=command_argument_count() 
      write(6,*) 'numArgs: ', numArgs

      IF (numArgs.EQ.1) THEN
        CALL get_command_argument(1, inpFilename)
      ELSE
          write(6,*) 'Too many or few few inp params.' 
          write(6,*) 'analysePDB inpFilename '
          STOP
      END IF
      write(6,*) 'Reading from: ',TRIM(inpFilename)

      !open the input file and create output files containing information about all the models in the input file.
      OPEN(7, FILE=inpFilename)
      OPEN(8, FILE='pdb.out')
      OPEN(9, FILE='temp.arrows')
      OPEN(13, FILE='pdb.xyz')
     
      !Read in information about how many models and how many building blocks per model
      READ(7,80) numModel
      READ(7,80) nBB
80    FORMAT(I4)

      !allocate arrays at the model level
      ALLOCATE(numArrowsPerModel(numModel))
      ALLOCATE(OCOG(numModel,3))
      ALLOCATE(cArray(nBB))
      ALLOCATE(CCOG1Array(nBB,1:3))
      ALLOCATE(CCOG2Array(nBB,1:3))
      ALLOCATE(BBCOG(nBB,1:3))
      ALLOCATE(BBX(nBB,1:3))
      ALLOCATE(BBY(nBB,1:3))
      ALLOCATE(BBZ(nBB,1:3))
      ALLOCATE(RMatU(numModel,1:3,1:3))

      !output basic info to stdout so user knows something is happening
      write(6,*) 'Analysing: ',numModel,' models containing: ',nBB,' building blocks each.'

      !get the information about the strands of interest: Up to the user to define which residues are 
      !within the beta strands of the amyloid backbone. 
      READ(7,81) nRb, nRe, nRS1b, nRS1e, nRS2b, nRS2e
      nRS1=nRS1e-nRS1b+1
      nRS2=nRS2e-nRS2b+1
81    FORMAT(I4,5(1X,I4))

!     write out basic information to standard out
      WRITE(6,81) nRb, nRe, nRS1b, nRS1e, nRS2b, nRS2e
72    FORMAT(I4)

!loop through each model containg in the PDB file.
      DO curModel=1,numModel,1

      !ouput the model specific headers for the arrows file and the xyz file
      write(9,*) 'replace' !temp.arrows
      write(9,*) 'Some Text' !temp.arrows
      Write(13,73) nBB*2 !'pdb.xyz'
73    FORMAT(I3)
      Write(13,*) ' Energy of minimum      1=     999.999 first found at step        0'  !pdb.xyz
 
      !Centre of gravity is cumulative for each model so set to zero initially.
      OCOG(curModel,1:3)=0
 
      !Initialise other running averages for this model
      aAModAv=0.0
      bAModAv=0.0
      cAModAv=0.0
      aRModAv=0.0
      bRModAv=0.0
      cRModAv=0.0
      dHorizModAv=0.0
      alphaModAv=0.0
      beta1ModAv=0.0
      beta2ModAv=0.0
      gamma1ModAv=0.0
      gamma2ModAv=0.0
      dVertModAv=0.0
 
!loop through building blocks and analyse each building block on its own
      DO curBB=1,nBB,1

         ALLOCATE(CPOS1(nRS1,3))
         ALLOCATE(CAPOS1(nRS1,3))
         ALLOCATE(CBPOS1(nRS1,3))
         ALLOCATE(CBPOS1Flag(nRS1))
         ALLOCATE(NPOS1(nRS1,3))
         ALLOCATE(OPOS1(nRS1,3))
         ALLOCATE(CPOS2(nRS2,3))
         ALLOCATE(CAPOS2(nRS2,3))
         ALLOCATE(CBPOS2(nRS2,3))
         ALLOCATE(CBPOS2Flag(nRS2))
         ALLOCATE(NPOS2(nRS2,3))
         ALLOCATE(OPOS2(nRS2,3))
         ALLOCATE(gamma1(nRS1))
         ALLOCATE(gamma2(nRS2))
 
         !counters to count the number of atoms in each strand of the building block
         nA1=0
         nA2=0

         curRS1=1
         curRS2=1

         !initialise position of centre of gravity (of the strand, not the building block)
         CCOG1=0
         CCOG2=0

         !loop through residues contributing to the current building block
         DO curR=nRb,nRe,1
 
!           WRITE(6,82) curR
82         FORMAT(I2)
 
           !read info about first residue of current building block
           READ(7,100) atomNum, atomType, residueName, chain, residueNum, insert, xDum, yDum, zDum, Occupancy, TempFact, SegmentId, ElementType
100   FORMAT ( 'ATOM  ',I5,1X,A4,1X,A3,1X,A1,I4,A1,3X,3F8.3,2F6.2,10X,A2,A2 )

!           WRITE(6,100) atomNum, atomType, residueName, chain, residueNum, insert, xDum, yDum, zDum, Occupancy, TempFact, SegmentId, ElementType

           !loop through the atoms of current residue. We don't know how many atoms are here so keep going until
           !residue number changes then bail out and move back a line in file. Scan for and Store useful information as we go.
           DO WHILE (residueNum.EQ.curR)
 
!           write(6,83) curR, residueNum
!83         FORMAT(I2,1X,I2, 1X, I2)

             !Adjust the string for correct comparison
             atomType=TRIM(ADJUSTL(atomType))

             !determine which strand the current residue belongs to
             IF ((curR.ge.nRS1b).AND.(curR.le.nRS1e)) THEN

!             write (6,84) curR
!84            Format ('Strand 1: ',I2)
 
!             write (6,86) atomType
!86           format ('atomType outer: ',A4)

             IF (atomType.EQ.'C') THEN
!              write (6,85),atomType
!85            FORMAT ('AtomType inner: ',A4)
                  CPOS1(curRS1,1)=xDum
                  CPOS1(curRS1,2)=yDum
                  CPOS1(curRS1,3)=zDum
                  CCOG1(1)=CCOG1(1)+xDum
                  CCOG1(2)=CCOG1(2)+yDum
                  CCOG1(3)=CCOG1(3)+zDum
                  nA1=nA1+1
              END IF
              IF (atomType.EQ.'CA') THEN
!                  write (6,85),atomType
                  CAPOS1(curRS1,1)=xDum
                  CAPOS1(curRS1,2)=yDum
                  CAPOS1(curRS1,3)=zDum
                  CCOG1(1)=CCOG1(1)+xDum
                  CCOG1(2)=CCOG1(2)+yDum
                  CCOG1(3)=CCOG1(3)+zDum
                  nA1=nA1+1
              END IF
              IF (atomType.EQ.'CB') THEN
                  CBPOS1(curRS1,1)=xDum
                  CBPOS1(curRS1,2)=yDum
                  CBPOS1(curRS1,3)=zDum
                  CBPOS1Flag(curRS1)=1
                  CCOG1(1)=CCOG1(1)+xDum
                  CCOG1(2)=CCOG1(2)+yDum
                  CCOG1(3)=CCOG1(3)+zDum
                  nA1=nA1+1
              END IF
              IF (atomType.EQ.'N') THEN
                  NPOS1(curRS1,1)=xDum
                  NPOS1(curRS1,2)=yDum
                  NPOS1(curRS1,3)=zDum
                  CCOG1(1)=CCOG1(1)+xDum
                  CCOG1(2)=CCOG1(2)+yDum
                  CCOG1(3)=CCOG1(3)+zDum
                  nA1=nA1+1
              END IF
              IF (atomType.EQ.'O') THEN
                  OPOS1(curRS1,1)=xDum
                  OPOS1(curRS1,2)=yDum
                  OPOS1(curRS1,3)=zDum
                  CCOG1(1)=CCOG1(1)+xDum
                  CCOG1(2)=CCOG1(2)+yDum
                  CCOG1(3)=CCOG1(3)+zDum
                  nA1=nA1+1
              END IF
             END IF

             !Check to see if residue is in the second strand
             IF ((curR.ge.nRS2b).AND.(curR.le.nRS2e)) THEN
              IF (atomType.EQ.'C') THEN
                  CPOS2(curRS2,1)=xDum
                  CPOS2(curRS2,2)=yDum
                  CPOS2(curRS2,3)=zDum
                  CCOG2(1)=CCOG2(1)+xDum
                  CCOG2(2)=CCOG2(2)+yDum
                  CCOG2(3)=CCOG2(3)+zDum
                  nA2=nA2+1
              END IF
              IF (atomType.EQ.'CA') THEN
                  CAPOS2(curRS2,1)=xDum
                  CAPOS2(curRS2,2)=yDum
                  CAPOS2(curRS2,3)=zDum
                  CCOG2(1)=CCOG2(1)+xDum
                  CCOG2(2)=CCOG2(2)+yDum
                  CCOG2(3)=CCOG2(3)+zDum
                  nA2=nA2+1
              END IF
              IF (atomType.EQ.'CB') THEN
                  CBPOS2(curRS2,1)=xDum
                  CBPOS2(curRS2,2)=yDum
                  CBPOS2(curRS2,3)=zDum
                  CBPOS2Flag(curRS2)=1
                  CCOG2(1)=CCOG2(1)+xDum
                  CCOG2(2)=CCOG2(2)+yDum
                  CCOG2(3)=CCOG2(3)+zDum
                  nA2=nA2+1
              END IF
              IF (atomType.EQ.'N') THEN
                  NPOS2(curRS2,1)=xDum
                  NPOS2(curRS2,2)=yDum
                  NPOS2(curRS2,3)=zDum
                  CCOG2(1)=CCOG2(1)+xDum
                  CCOG2(2)=CCOG2(2)+yDum
                  CCOG2(3)=CCOG2(3)+zDum
                  nA2=nA2+1
              END IF
              IF (atomType.EQ.'O') THEN
                  OPOS2(curRS2,1)=xDum
                  OPOS2(curRS2,2)=yDum
                  OPOS2(curRS2,3)=zDum
                  CCOG2(1)=CCOG2(1)+xDum
                  CCOG2(2)=CCOG2(2)+yDum
                  CCOG2(3)=CCOG2(3)+zDum
                  nA2=nA2+1
              END IF
             END IF

             !read next line in file - Have to add a dummy line at end of input file so there is a line to read; 
             !I repeat the last line in the data file but increment residue nyumber by one to force this loop to bail out
             READ(7,100) atomNum, atomType, residueName, chain, residueNum, insert, xDum, yDum, zDum, Occupancy, TempFact, SegmentId, ElementType

           !end while loop for looping through the atoms of current residue
           END DO

           !have just finished a residue so increment the output array index depending in which residue we have just finished
           IF ((curR.ge.nRS2b).AND.(curR.le.nRS2e)) THEN
              curRS2=curRS2+1
           END IF
           IF ((curR.ge.nRS1b).AND.(curR.le.nRS1e)) THEN
              curRS1=curRS1+1
           END IF

           !would have read one record too far so backspace one record in the file
           BACKSPACE 7 

         !end the do loop which loops through the right number of residues for each building block
         END DO

!         DO curR=1,nRS1,1
!            write(6,88) CPOS1(curR,1), CAPOS1(curR,1), CBPOS1(curR,1), NPOS1(curR,1), OPOS1(curR,1)
!88          format('C: ',f8.3, ', CA: ',f8.3,', CB: ', f8.3,', N: ',f8.3, ', O: ',f8.3)
!         END DO
!         DO curR=1,nRS2,1
!            write(6,88) CPOS2(curR,1), CAPOS2(curR,1), CBPOS2(curR,1), NPOS2(curR,1), OPOS2(curR,1)
!         END DO


         !Now have the entire data for a single building block loaded and split into two strands in two commensurate set of arrays.

         !Find d and X-axis - vector between centres of gravity of each strand
         !for each atom we stored in the previous loops we added it's co-ordinates
         !to the cumulative sum. Now divide by number of atoms stored to get centre of gravity 
         !(assume all atoms have same mass and get the positional centroid, rather than true centre of mass).
 
         CCOG1(1:3)=CCOG1(1:3)/nA1
         CCOG2(1:3)=CCOG2(1:3)/nA2
         xAxis=CCOG2-CCOG1
         d=SQRT(xAxis(1)**2+xAxis(2)**2+xAxis(3)**2)
         xAxis=xAxis/d

         !set up a format for outputting 3 floats to file.
84       FORMAT (F8.3,2(1X,F8.3))

         !Find b1 and b2; 
         !i.e. the vector between each equivalent atom in the end two residues for an  odd number
         !or the vector between one end point and the penultimate residue at the other end for an even number of residues

         !define a b1 for each atom type for strand 1 
         IF (MODULO(nRS1,2).EQ.0) THEN
           !dealing with odd number of residues
           b1C=CPOS1(nRS1-1,:)-CPOS1(1,:)
           b1CA=CAPOS1(nRS1-1,:)-CAPOS1(1,:)
           b1N=NPOS1(nRS1-1,:)-NPOS1(1,:)
           b1O=OPOS1(nRS1-1,:)-OPOS1(1,:)
         ELSE
           !dealing with even numbr of
           b1C=CPOS1(nRS1,:)-CPOS1(1,:)
           b1CA=CAPOS1(nRS1,:)-CAPOS1(1,:)
           b1N=NPOS1(nRS1,:)-NPOS1(1,:)
           b1O=OPOS1(nRS1,:)-OPOS1(1,:)
         END IF
         IF (MODULO(nRS2,2).EQ.0) THEN
           !dealing with odd number of residues
           b2C=CPOS2(nRS2-1,:)-CPOS2(1,:)
           b2CA=CAPOS2(nRS2-1,:)-CAPOS2(1,:)
           b2N=NPOS2(nRS2-1,:)-NPOS2(1,:)
           b2O=OPOS2(nRS2-1,:)-OPOS2(1,:)
         ELSE
           !dealing with even number of
           b2C=CPOS2(nRS2,:)-CPOS2(1,:)
           b2CA=CAPOS2(nRS2,:)-CAPOS2(1,:)
           b2N=NPOS2(nRS2,:)-NPOS2(1,:)
           b2O=OPOS2(nRS2,:)-OPOS2(1,:)
         END IF

         !compute b vectors as the average of all these vectors
         b1(1:3)=(b1C(1:3)+b1CA(1:3)+b1N(1:3)+b1O(1:3))/ 4.0
         b1d=SQRT(b1(1)**2+b1(2)**2+b1(3)**2)
         b1=b1/b1d
         b2=(b2C+b2CA+b2N+b2O)/4.0
         b2d=SQRT(b2(1)**2+b2(2)**2+b2(3)**2)
         b2=b2/b2d
 
         !compute y and zAxes and beta1
         XDotB=xAxis(1)*b1(1)+xAxis(2)*b1(2)+xAxis(3)*b1(3)
         beta1=(ACOS(XDotB)-pi/2)*180/pi
         yAxis(1:3)=b1(1:3) - xDotB*xAxis(1:3)
         yAxis=yAxis/SQRT(yAxis(1)**2+yAxis(2)**2+yAxis(3)**2)
         CALL CROSSOPT(xAxis, yAxis, zAxis, 1)
 
         !Check for sign of zAxis z component in lab frame. If it's pointing down, then flip b1 and repeat operation; purely aesthetic.
         IF (zAxis(3).lt.0) THEN
             b1(1:3)=-1*b1(1:3)
             XDotB=xAxis(1)*b1(1)+xAxis(2)*b1(2)+xAxis(3)*b1(3)
             beta1=(ACOS(XDotB)-pi/2)*180/pi
             yAxis(1:3)=b1(1:3) - xDotB*xAxis(1:3)
             yAxis=yAxis/SQRT(yAxis(1)**2+yAxis(2)**2+yAxis(3)**2)
             CALL CROSSOPT(xAxis, yAxis, zAxis,1)
         END IF

         !Compute position to start drawing the a b1 vector of length 2d so it passes through the COG of ellipsoid 1 at it mid-point       
         SPOS1(1:3)=CCOG1(1:3)-d*b1(1:3)
 
         ! construct a1  - line of nodes of ellipsoid 1 prior to gamma rotation.
         CALL CROSSOPT(b1,zAxis,a1,1)
         !construct c1 - a1 x b1
         CALL CROSSOPT(a1,b1,c1,1)
 
         !compute gamma1 
         nB1=0
         gamma1Av=0
         a1dSum=0
         c1dSum=0
         DO curR=1,nRS1,1
          !Check we have a CB
          IF (CBPOS1Flag(curR).EQ.1) THEN
 
             !compute vector
             CAB(:)=CBPOS1(curR,:)-CAPOS1(curR,:)

             !normalise CAB vector
             CAB(:)=CAB(:)/SQRT(CAB(1)**2+CAB(2)**2+CAB(3)**2)

             !project vector on b1
             CABb1=b1(1)*CAB(1)+b1(2)*CAB(2)+b1(3)*CAB(3)

             !extract CABvector in plane perpendicular to b1 and normalise
             CABac(1:3)=CAB(1:3)-CABb1*b1(1:3)
             CABac(:)=CABac(:)/SQRT(CABac(1)**2+CABac(2)**2+CABac(3)**2)

             !if required plot CAB and CABac at each point.
             numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
             write(9,120) CAPOS1(curR,1),CAPOS1(curR,2),CAPOS1(curR,3),CAB(1)+CAPOS1(curR,1),CAB(2)+CAPOS1(curR,2),CAB(3)+CAPOS1(curR,3),400.0,CylWidth,ConeWidth
             numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
             write(9,120) CAPOS1(curR,1),CAPOS1(curR,2),CAPOS1(curR,3),CABac(1)+CAPOS1(curR,1),CABac(2)+CAPOS1(curR,2),CABac(3)+CAPOS1(curR,3),800.0,CylWidth,ConeWidth

             !project CABac on to a1 and z
             CABaca1=CABac(1)*a1(1)+CABac(2)*a1(2)+CABac(3)*a1(3)
             CABacz1=CABac(1)*zAxis(1)+CABac(2)*zAxis(2)+CABac(3)*zAxis(3)

             !compute angle of CABac with a1 axis.  COS(gamma1)=CABac.a1
             gamma1(curR)=ATAN2(CABacz1,CABaca1)

             !Check range of gamma1 lies with in -pi/2 <= gamma1 <= pi/2 (!takes into account when CAB is on other side of strand)
             IF (gamma1(curR).gt.pi/2) THEN
                gamma1(curR)=gamma1(curR)-pi
             END IF
             IF (gamma1(curR).lt.-pi/2) THEN
                gamma1(curR)=gamma1(curR)+pi
             END IF

             !compute cumulative sum of gamma to average later.
             gamma1Av=gamma1Av+gamma1(curR)*180/pi

             !Find the perpendicular distance of cB from the line through COG llel to b1.
             COG1CB(1:3)=CBPOS1(curR,:)-CCOG1(1:3) !vector from COG to CB
             b1CBp(1:3)=COG1CB(1:3)-(COG1CB(1)*b1(1)+COG1CB(2)*b1(2)+COG1CB(3)*b1(3))*b1(1:3) !vector from CB to line through COG llel to b1
             a1d=SQRT(b1CBp(1)**2+b1CBp(2)**2+b1CBp(3)**2) !magnitude of that vector
             a1dSum=a1dSum+a1d !accumulate the ds to compute average distance of CB - probably a reasonable estimate of the repulsive ellipsoid.

             !count the number of CB atoms for the averages
             nB1=nB1+1
           END IF
           !Find the perpendicular distance of O from the line through COG llel to b1.
           COG1O(1:3)=OPOS1(curR,:)-CCOG1(1:3) !vector from COG to  O
           b1Op(1:3)=COG1O(1:3)-(COG1O(1)*b1(1)+COG1O(2)*b1(2)+COG1O(3)*b1(3))*b1(1:3) !shortest vector to O from line through COG llel to b1
           c1d=SQRT(b1Op(1)**2+b1Op(2)**2+b1Op(3)**2) !magnitude of that vector
           c1dSum=c1dSum+c1d !accumulate the ds to compute average distance of O from b1  - probably a reasonable estimate of the repulsive ellipsoid.
        END DO
 
        !compute average gamma1 from CA-CB vectors, double half lengths of ellipsoid for output to .xyz file
        gamma1Av=gamma1Av/nB1
        c1d=2.0D0*c1dSum/nRS1
        a1d=2.0D0*a1dSum/nB1 

        !rotate a1 out of the xy plane by an angle gamma1 in the c1-a1 plane, 
        !and rotate the c1 axis by an amount gamma in the c1-a1 plane.
        a1g(1:3)=SIN(gamma1Av*pi/180)*c1(1:3)+COS(gamma1Av*pi/180)*a1(1:3)
        a1g(1:3)=a1g(1:3)/SQRT(a1g(1)**2+a1g(2)**2+a1g(3)**2)
        c1g(1:3)=COS(gamma1Av*pi/180)*c1(1:3)-SIN(gamma1Av*pi/180)*a1(1:3)
        c1g(1:3)=c1g(1:3)/SQRT(c1g(1)**2+c1g(2)**2+c1g(3)**2)
 
 
        !repeat for second ellipsoid as required
        SPOS2(1:3)=CCOG2(1:3)-d*b2(1:3)

        !compute projection of b2 on z-axis
        b2DotZ=b2(1)*zAxis(1)+b2(2)*zAxis(2)+b2(3)*zAxis(3)

        !compute projection of b2 on XY plane
        b2XY(1:3)=b2(1:3)-b2DotZ*zAxis(1:3)

        !compute projection b2 on x and y
        b2XYdotX=b2XY(1)*xAxis(1)+b2XY(2)*xAxis(2)+b2XY(3)*xAxis(3)
        b2XYdotY=b2XY(1)*yAxis(1)+b2XY(2)*yAxis(2)+b2XY(3)*yAxis(3)

        !compute beta2
        beta2=-ATAN2(b2XYdotX, b2XYdotY)

        !check for cases in which b2 is pointing the wrong way; flip it and recompute.
        IF (abs(beta2).gt.pi/2) THEN
            !flip the b2 vector
            b2(1:3)=-1*b2(1:3)
          
            !repeat for second ellipsoid as required
            SPOS2(1:3)=CCOG2(1:3)-d*b2(1:3)

            !compute projection of b2 on z-axis
            b2DotZ=b2(1)*zAxis(1)+b2(2)*zAxis(2)+b2(3)*zAxis(3)

            !compute projection of b2 on XY plane
            b2XY(1:3)=b2(1:3)-b2DotZ*zAxis(1:3)

            !compute projection b2 on x and y
            b2XYdotX=b2XY(1)*xAxis(1)+b2XY(2)*xAxis(2)+b2XY(3)*xAxis(3)
            b2XYdotY=b2XY(1)*yAxis(1)+b2XY(2)*yAxis(2)+b2XY(3)*yAxis(3)

            !compute beta2
            beta2=-ATAN2(b2XYdotX, b2XYdotY)
        END IF
        !convert beta2 to degrees
        beta2=beta2*180/pi


        !compute alpha have already resolved the b2 flip so shouldn't need a second one.
        alpha=ATAN2(b2DotZ, SQRT(b2XY(1)**2+b2XY(2)**2+b2XY(3)**2))*180/pi

        !compute vector of line of nodes prior to gamma rotation
        ! (a axis = cross(b2 ,zAxis). (won't work for alpha=pi/2, but I'll take the chance).
        CALL CROSSOPT(b2,zAxis,a2,1)
        !construct c2 - a2 x b2
        CALL CROSSOPT(a2,b2,c2,1)
 
        !compute gamma 2
        nB2=0
        gamma2Av=0
        a2dSum=0
        c2dSum=0 
       !repeat for gamma2
        DO curR=1,nRS2,1
          !Check we have a CB
          IF (CBPOS2Flag(curR).EQ.1) THEN

             !compute vector
             CAB(1:3)=CBPOS2(curR,:)-CAPOS2(curR,:)

             !normalise CAB vector
             CAB(1:3)=CAB(1:3)/SQRT(CAB(1)**2+CAB(2)**2+CAB(3)**2)

             !project vector on b2
             CABb2=b2(1)*CAB(1)+b2(2)*CAB(2)+b2(3)*CAB(3)
 
             !extract CABvector in plane perpendicular to b2 and normalise
             CABac(1:3)=CAB(1:3)-CABb2*b2(1:3)
             CABac=CABac/SQRT(CABac(1)**2+CABac(2)**2+CABac(3)**2)

             !if required plot CAB and CABac
             numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
             write(9,120) CAPOS2(curR,1),CAPOS2(curR,2),CAPOS2(curR,3),CAB(1)+CAPOS2(curR,1),CAB(2)+CAPOS2(curR,2),CAB(3)+CAPOS2(curR,3),400.0,CylWidth,ConeWidth
             numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
             write(9,120) CAPOS2(curR,1),CAPOS2(curR,2),CAPOS2(curR,3),CABac(1)+CAPOS2(curR,1),CABac(2)+CAPOS2(curR,2),CABac(3)+CAPOS2(curR,3),800.0,CylWidth,ConeWidth

             !project CABac on to a2 and c2
             CABaca2=CABac(1)*a2(1)+CABac(2)*a2(2)+CABac(3)*a2(3)
             CABacc2=CABac(1)*c2(1)+CABac(2)*c2(2)+CABac(3)*c2(3)

             !compute angle of CABac with a2 axis. 
             gamma2(curR)=ATAN2(CABacc2,CABaca2)

             !Check range of gamma2 lies with in -pi/2 <= gamma2 <= pi/2 (!takes into account when CAB is on other side of strand)
             IF (gamma2(curR).gt.pi/2) THEN
                gamma2(curR)=gamma2(curR)-pi
             END IF
             IF (gamma2(curR).lt.-pi/2) THEN
                gamma2(curR)=gamma2(curR)+pi
             END IF

             !compute cumulative sum of gamma to average later.
             gamma2Av=gamma2Av+gamma2(curR)*180/pi

             !Find the perpendicular distance of cB from the line through COG llel to b2.
             COG2CB(1:3)=CBPOS2(curR,:)-CCOG2(1:3) !vector from COG to CB
             b2CBp(1:3)=COG2CB(1:3)-(COG2CB(1)*b2(1)+COG2CB(2)*b2(2)+COG2CB(3)*b2(3))*b2(1:3) !vector from CB to line through COG llel to b2
             a2d=SQRT(b2CBp(1)**2+b2CBp(2)**2+b2CBp(3)**2) !magnitude of that vector
             a2dSum=a2dSum+a2d !accumulate the ds to compute average distance of CB - probably a reasonable estimate of the repulsive ellipsoid.
 
             !count number of residues with a CB
             nB2=nB2+1
          END IF
          !Find the perpendicular distance of O from the line through COG llel to b2.
          COG2O(1:3)=OPOS2(curR,:)-CCOG2(1:3) !vector from COG to  O
          b2Op(1:3)=COG2O(1:3)-(COG2O(1)*b2(1)+COG2O(2)*b2(2)+COG2O(3)*b2(3))*b2(1:3) !shortest vector to O from line through COG llel to b2
          c2d=SQRT(b2Op(1)**2+b2Op(2)**2+b2Op(3)**2) !magnitude of that vector
          c2dSum=c2dSum+c2d
        END DO

        !compute means
        gamma2Av=gamma2Av/nB2
        a2d=2.0D0*a2dSum/nB2
        c2d=2.0D0*c2dSum/nRS2

        !rotate a2 out of the xy plane by an angle gamma2 in the c2-a2 plane, 
        !and rotate the c2 axis by an amount gamma2 in the c2-a2 plane.
        a2g(1:3)=SIN(gamma2Av*pi/180)*c2(1:3)+COS(gamma2Av*pi/180)*a2(1:3)
        a2g(1:3)=a2g(1:3)/SQRT(a2g(1)**2+a2g(2)**2+a2g(3)**2)
        c2g(1:3)=COS(gamma2Av*pi/180)*c2(1:3)-SIN(gamma2Av*pi/180)*a2(1:3)
        c2g(1:3)=c2g(1:3)/SQRT(c2g(1)**2+c2g(2)**2+c2g(3)**2)

! output all the vectors of interest - the principle axes, the body axes of each ellipsoid and a few construction vectors in the .arr file
 
!       Basis axes in red (100) - line of centres and the y and z axes repeated at each ellipsoid.
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) CCOG1(1), CCOG1(2),CCOG1(3),CCOG2(1),CCOG2(2),CCOG2(3),100.0,CylWidth, ConeWidth
120     format ('a',f8.3,8(1x,f8.3))
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) CCOG1(1),CCOG1(2),CCOG1(3),d*yAxis(1)+CCOG1(1),d*yAxis(2)+CCOG1(2),d*yAxis(3)+CCOG1(3),100.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CCOG2(1),CCOG2(2),CCOG2(3),d*yAxis(1)+CCOG2(1),d*yAxis(2)+CCOG2(2),d*yAxis(3)+CCOG2(3),100.0,CylWidth, ConeWidth
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) CCOG1(1),CCOG1(2),CCOG1(3),d*zAxis(1)+CCOG1(1),d*zAxis(2)+CCOG1(2),d*zAxis(3)+CCOG1(3),100.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CCOG2(1),CCOG2(2),CCOG2(3),d*zAxis(1)+CCOG2(1),d*zAxis(2)+CCOG2(2),d*zAxis(3)+CCOG2(3),100.0,CylWidth, ConeWidth


!       Interim construction vectors painted in white. 

        !The projection of b2 on the xy plane.
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CCOG2(1),CCOG2(2),CCOG2(3),d*b2XY(1)+CCOG2(1),d*b2XY(2)+CCOG2(2),d*b2XY(3)+CCOG2(3),400.0,CylWidth, ConeWidth

!        Atom to atom vectors along the beta strands for first ellipsoid
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CPOS1(1,1),CPOS1(1,2),CPOS1(1,3),b1C(1)+CPOS1(1,1),b1C(2)+CPOS1(1,2),b1C(3)+CPOS1(1,3),400.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CAPOS1(1,1),CAPOS1(1,2),CAPOS1(1,3),b1CA(1)+CAPOS1(1,1),b1CA(2)+CAPOS1(1,2),b1CA(3)+CAPOS1(1,3),400.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) NPOS1(1,1),NPOS1(1,2),NPOS1(1,3),b1N(1)+NPOS1(1,1),b1N(2)+NPOS1(1,2),b1N(3)+NPOS1(1,3),400.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) OPOS1(1,1),OPOS1(1,2),OPOS1(1,3),b1O(1)+OPOS1(1,1),b1O(2)+OPOS1(1,2),b1O(3)+OPOS1(1,3),400.0,CylWidth, ConeWidth

!        Atom to atom vectors along the beta strands for second ellipsoid
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CPOS2(1,1),CPOS2(1,2),CPOS2(1,3),b2C(1)+CPOS2(1,1),b2C(2)+CPOS2(1,2),b2C(3)+CPOS2(1,3),400.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CAPOS2(1,1),CAPOS2(1,2),CAPOS2(1,3),b2CA(1)+CAPOS2(1,1),b2CA(2)+CAPOS2(1,2),b2CA(3)+CAPOS2(1,3),400.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) NPOS2(1,1),NPOS2(1,2),NPOS2(1,3),b2N(1)+NPOS2(1,1),b2N(2)+NPOS2(1,2),b2N(3)+NPOS2(1,3),400.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) OPOS2(1,1),OPOS2(1,2),OPOS2(1,3),b2O(1)+OPOS2(1,1),b2O(2)+OPOS2(1,2),b2O(3)+OPOS2(1,3),400.0,CylWidth, ConeWidth
 

!      the a1 and c1 axes prior to rotation
!       numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CCOG1(1),CCOG1(2),CCOG1(3),d*a1(1)+CCOG1(1),d*a1(2)+CCOG1(2),d*a1(3)+CCOG1(3),400.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CCOG1(1),CCOG1(2),CCOG1(3),d*c1(1)+CCOG1(1),d*c1(2)+CCOG1(2),d*c1(3)+CCOG1(3),400.0,CylWidth, ConeWidth
!       a2 and c2 axis 
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CCOG2(1),CCOG2(2),CCOG2(3),d*a2(1)+CCOG2(1),d*a2(2)+CCOG2(2),d*a2(3)+CCOG2(3),400.0,CylWidth, ConeWidth
!        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
!        write(9,120) CCOG2(1),CCOG2(2),CCOG2(3),d*c2(1)+CCOG2(1),d*c2(2)+CCOG2(2),d*c2(3)+CCOG2(3),400.0,CylWidth, ConeWidth
 

!       NOw plot the body axes in blue

!       average atom to atom vector - reasonable approximation to b1 and b2
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) SPOS1(1),SPOS1(2),SPOS1(3),2*d*b1(1)+SPOS1(1),2*d*b1(2)+SPOS1(2),2*d*b1(3)+SPOS1(3),800.0,CylWidth, ConeWidth
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) SPOS2(1),SPOS2(2),SPOS2(3),2*d*b2(1)+SPOS2(1),2*d*b2(2)+SPOS2(2),2*d*b2(3)+SPOS2(3),800.0,CylWidth, ConeWidth

!       a1 and c1 axis 
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) CCOG1(1),CCOG1(2),CCOG1(3),d*a1g(1)+CCOG1(1),d*a1g(2)+CCOG1(2),d*a1g(3)+CCOG1(3),800.0,CylWidth, ConeWidth
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) CCOG1(1),CCOG1(2),CCOG1(3),d*c1g(1)+CCOG1(1),d*c1g(2)+CCOG1(2),d*c1g(3)+CCOG1(3),800.0,CylWidth, ConeWidth
!       a2 and c2 axis 
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) CCOG2(1),CCOG2(2),CCOG2(3),d*a2g(1)+CCOG2(1),d*a2g(2)+CCOG2(2),d*a2g(3)+CCOG2(3),800.0,CylWidth, ConeWidth
        numArrowsPerModel(curModel)=numArrowsPerModel(curModel)+1
        write(9,120) CCOG2(1),CCOG2(2),CCOG2(3),d*c2g(1)+CCOG2(1),d*c2g(2)+CCOG2(2),d*c2g(3)+CCOG2(3),800.0,CylWidth, ConeWidth
 
        !form a rotation matrix from the body axes
        Rmat1(1:3,1)=a1g
        Rmat1(1:3,2)=b1
        Rmat1(1:3,3)=c1g
        Rmat2(1:3,1)=a2g
        Rmat2(1:3,2)=b2
        Rmat2(1:3,3)=c2g

        CALL RMAT2AA(Rmat1,p1)
        CALL RMAT2AA(Rmat2,p2)
        !write out the .xyz file for ease of plotting im VMD or makemol
        WRITE(13,'(a5,2x,3f20.10,2x,a8,12f15.8,2x,a11,3f15.8)') 'O',CCOG1,&
                    'ellipse',a1d, b1d, c1d, Rmat1(1,1:3), Rmat1(2,1:3), Rmat1(3,1:3),&
                    'atom_vector', p1
        WRITE(13,'(a5,2x,3f20.10,2x,a8,12f15.8,2x,a11,3f15.8)') 'O',CCOG2,&
                    'ellipse',a2d, b2d, c2d, Rmat2(1,1:3), Rmat2(2,1:3), Rmat2(3,1:3),&
                    'atom_vector', p2


        !Compute overall centre of gravity for building block and remember
        BBCOG(curBB,1:3)=(CCOG1(1:3)+CCOG2(1:3))/2.0D0

        !Remember XY and Z axes of building block to compute the coords file for the building blocks
        BBX(curBB,1:3)=xAxis
        BBY(curBB,1:3)=yAxis
        BBZ(curBB,1:3)=zAxis

        !compute cumulative COG to get Overall COG for model
        OCOG(curModel,1:3)=OCOG(curModel,1:3)+BBCOG(curBB,1:3)

        !write out basic result for each building block in the .out file
        write(8,95) curModel, curBB, d,alpha, beta1, beta2, gamma1Av, gamma2Av
95      FORMAT(I2,1X,I2,1X,6F8.3)


        !Store the output values for later averaging. - variety of formulae for each type.
        CCOG1Array(curBB,1:3)=CCOG1(1:3)
        CCOG2Array(curBB,1:3)=CCOG2(1:3)
        dHorizModAv=dHorizModAv+d
        alphaModAv=alphaModAv+alpha
        beta1ModAv=beta1ModAv+beta1
        beta2ModAv=beta2ModAv+beta2
        gamma1ModAv=gamma1ModAv+gamma1Av
        gamma2ModAv=gamma2ModAv+gamma2Av
        aAModAv=aAModAv+d - 1.3D0*(a1d+a2d)/2.0D0 !invented rule yields initial attractive range:  d - 2*repulsive range of other ellipsoid.
        bAModAv=bAModAv+0.04D0*(b1d+b2d)/2.0D0 !attractive end force is 4% of the repulsive one.
        cAModAv=cAModAv+0.95D0*(c1d+c2d)/2.0D0 !95% of repulsive force
!        cArray(curBB)=(c1d+c2d)/2.0D0 !this is taken to be the same as the repulsive force
        aRModAv=aRModAv+((a1d+a2d)/2.0D0)
        bRModAv=bRModAv+((b1d+b2d)/2.0D0)
        cRModAv=cRModAv+((c1d+c2d)/2.0D0)

        DEALLOCATE(CPOS1)
        DEALLOCATE(CAPOS1)
        DEALLOCATE(CBPOS1)
        DEALLOCATE(CBPOS1Flag)
        DEALLOCATE(NPOS1)
        DEALLOCATE(OPOS1)
        DEALLOCATE(CPOS2)
        DEALLOCATE(CAPOS2)
        DEALLOCATE(CBPOS2)
        DEALLOCATE(CBPOS2Flag)
        DEALLOCATE(NPOS2)
        DEALLOCATE(OPOS2)
        DEALLOCATE(gamma1)
        DEALLOCATE(gamma2)

      END DO !end of buildingblock loop
 
      !compute average centre of gravity for current model from cumulative sum
      OCOG(curModel,1:3)=OCOG(curModel,1:3)/nBB

!********** coords file with information on how the building blocks are arranged in the PDB in centred and uncentered formats
!********** combined with a pysites file can be used with optim to derive as, bs and cs that give a structure as close as possible 
!********** to the original when minimised.

      !create a directory for the current model if it doesn't already exist
      WRITE(fileNumStr,136) curModel
136   FORMAT('model',I2.2)
 
      fileNumStr = ADJUSTL(TRIM(fileNumStr))
      outFilename = ADJUSTL(TRIM(fileNumStr))//ADJUSTL(TRIM('/.'))
      INQUIRE(FILE=outFilename, EXIST=UNITOK)
      IF (UNITOK==0) THEN
         CALL system('mkdir '//ADJUSTL(TRIM(fileNumStr)))
      END IF


! output the finish file - centres and orientations of each building block
      outFilename = ADJUSTL(TRIM(fileNumStr))//ADJUSTL(TRIM('/finish'))
      OPEN(14, FILE=outFilename)
      !loop through the building blocks outputting their position and orientation in the model, 
      !one in the lab frame, one in centralised and one in centralised and reoriented coords
      DO curBB=1,nBB,1
        WRITE(14,94) BBCOG(curBB,1), BBCOG(curBB,2), BBCOG(curBB,3)
94      FORMAT(F8.3,1X,F8.3,1X,F8.3)
      END DO

      !having written the COGS, now do the rotations in angle axis formulation
      DO curBB=1,nBB,1
        !Compute the lab frame rotation matrix of each building block
        RMat(1:3,1)=BBX(curBB,1:3)
        RMat(1:3,2)=BBY(curBB,1:3)
        RMat(1:3,3)=BBZ(curBB,1:3)

        ! convert the rotation matrices to angle axis vectors.  
        CALL RMAT2AA(RMat,p1)
        WRITE(14,94) p1(1), p1(2), p1(3)
      END DO
      CLOSE(14)


! **********  The model parameters for the building block.
! Encodes the relative position and orientation of the ellipsoids in the building block and starting values for the as,bs and cs.
! Run this file through genCoords to generate a pysites.xyz file and a coords file with nBB building blocks arranged in a vertical stack.
! This coords file can be replaced with BBCoords or BBCoordsC to create a starting point very close the the PDB

      !Compute distance between strands and c attractive
      DO curBB=1,nBB-1,1
         dCCOG1=CCOG1Array(curBB+1,1:3)-CCOG1Array(curBB,1:3)
         dCCOG2=CCOG2Array(curBB+1,1:3)-CCOG2Array(curBB,1:3)
         
         dVertModAv=dVertModAv+(SQRT(dCCOG1(1)**2+dCCOG1(2)**2+dCCOG1(3)**2)+SQRT(dCCOG2(1)**2+dCCOG2(2)**2+dCCOG2(3)**2))/2.0D0
      END DO
      dVertModAv=dVertModAv/(nBB-1)

      !divide sums to get averages for each parameter to yield overall value for current model
      dHorizModAv=dHorizModAv/nBB
      alphaModAv=alphaModAv/nBB
      beta1ModAv=beta1ModAv/nBB
      beta2ModAv=beta2ModAv/nBB
      gamma1ModAv=gamma1ModAv/nBB
      gamma2ModAv=gamma2ModAv/nBB
      aAModAv=aAModAv/nBB
      bAModAv=bAModAv/nBB
      cAModAv=cAModAv/nBB      
!cArray(curBB)=(c1d+c2d)/2.0D0 !this should be distance between COGS - repulsive of next ellipsoid up so need to store repulsive info between loops.
      aRModAv=aRModAv/nBB
      bRModAv=bRModAv/nBB
      cRModAv=cRModAv/nBB

      !Create a .model file for each model
      write(fileNumStr,136) curModel
      outFilename = ADJUSTL(TRIM(fileNumStr))//'/ellipsoid.model'

      !output the averages for each model in its own .model file
      OPEN(14, FILE=outFilename)
      WRITE(14,137) nBB
137   FORMAT('angle 13',I2,' 2 1.0 1.0')
      WRITE(14,65) aRModAv
65    FORMAT(F8.3)
      WRITE(14,65) bRModAv
      WRITE(14,65) cRModAv
      WRITE(14,65) aAModAv
      WRITE(14,65) bAModAv
      WRITE(14,65) cAModAv
      WRITE(14,65) dHorizModAv
      WRITE(14,65) alphaModAv
      WRITE(14,65) beta1ModAv
      WRITE(14,65) beta2ModAv
      WRITE(14,65) gamma1ModAv
      WRITE(14,65) gamma2ModAv
      WRITE(14,65) dVertModAv/cRModAv
      CLOSE(14)

      END DO !end of model level loop
      close(7)
      close(8)
      close(9)
      close(10)
      close(11)
      close(12)
      CLOSE(13)

! *********  Rewrite the arrows file now we know how many arrows there are per model. Could do this in centred coords too. But we're not.
      OPEN(9, FILE='temp.arrows')
      OPEN(10, FILE='pdb.arr')

      curModel=1
      continueFlag=1
      DO WHILE (continueFlag==1)
        READ(9,'(A)', IOSTAT=reason) buffer
       
        IF (reason==0) THEN
          IF (TRIM(ADJUSTL(buffer))=='replace') THEN
            WRITE(10,119) numArrowsPerModel(curModel)
            curModel=curModel+1
119   FORMAT(I3)
          ELSE
            WRITE(10,*) TRIM(ADJUSTL(buffer))
          END IF
        ELSE
          continueFlag=0
        END IF
      END DO
      CLOSE(9)
      CLOSE(10)

      CALL system('rm temp.*')

! deallocate arrays allocated at the model level
      DEALLOCATE(cArray)
      DEALLOCATE(OCOG)
      DEALLOCATE(BBCOG)
      DEALLOCATE(BBX)
      DEALLOCATE(BBY)
      DEALLOCATE(BBZ)
      DEALLOCATE(CCOG1Array)
      DEALLOCATE(CCOG2Array)
      DEALLOCATE(numArrowsPerModel)
      DEALLOCATE(RMatU)
     END

INCLUDE '/home/cjf41/Dropbox/Source/fortran/crossOpt.f90'
INCLUDE '/home/cjf41/Dropbox/Source/fortran/normal.f90'
INCLUDE '/home/cjf41/Dropbox/Source/fortran/cpoa.f90'
INCLUDE '/home/cjf41/Dropbox/Source/fortran/rmat2aa.f90'
