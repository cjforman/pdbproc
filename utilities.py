    def readTextFile(self,filename):
	    #read line data in from file
        try:
            vst = open(filename, 'r')
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            raise Exception, "Unable to open input file: "+filename
        lines = vst.readlines()
        vst.close()
        return lines

    def writeTextFile(lines,filename):
	    #write line data to file
	    try:
	        vst = open(filename, 'w')
	    except:
	        raise Exception, "Unable to open output file: "+filename
	    
	    for line in lines:
	        a=line
	        vst.write(a)
	    vst.close()
	    return
