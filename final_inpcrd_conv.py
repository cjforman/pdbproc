if __name__ == "__main__":
    input_name = "points.final"
    output_name = "coords.inpcrd"
    print input_name
    f_in = open(input_name,'r')
    f_out = open(output_name,'w')
    data_raw = f_in.readlines()
    f_in.close()    
 
    data = [ float(coord) for line in data_raw for coord in line.split()]
    coords_string = ["{:12.7f}".format(a) for a in data]
    f_out.write("Default Name\n")
    f_out.write(str(len(data)/3)+'\n')
    for i in range(0,len(data),6):
        try:
            f_out.write(coords_string[i])
            f_out.write(coords_string[i+1])
            f_out.write(coords_string[i+2])
            f_out.write(coords_string[i+3])
            f_out.write(coords_string[i+4])
            f_out.write(coords_string[i+5]+'\n')
        except IndexError:
            print "got to the end"
    f_out.close()    
