import csv

# python program to add metadata and dummy columns
# for use in PCA analysis

input_file_prism = 'processed.cleveland.data'
output_file = 'numbered_cleveland_data.txt'

output_file_ptr  = open(output_file, 'w')

iCount = 1

output_file_ptr.write('age,sex,cp,trestbps,(chol),(fbs),(restecg),(thalach),(exang),(oldpeak),(slope),(ca),(thal),(num)' + '\n')

with open(input_file_prism, 'r') as input_file_ptr_prism:
    
    reader = csv.reader(input_file_ptr_prism, delimiter = ',')
    for rRow in reader:

        temp_str =  ','.join(rRow)            
        output_file_ptr.write(str(iCount) + ',' + temp_str + '\n')
        
        iCount = iCount + 1 


output_file_ptr.close()        

if __name__ == "__main__":
    import sys
    #fasta_read(int(sys.argv[1]))
