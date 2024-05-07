# file="WF_hi_v1.3"
# arr=file.split("_")

# print(file)

# print( '_'.join(file.split("_")[:-1]) )

# f1=f"{'_'.join(file.split('_')[:-1])}_{file.split('_')[-1].split('.')[0]}.{1+int(file.split('_')[-1].split('.')[1])}"
# f2=f"{'_'.join(file.split('_')[:-1])}_{file.split('_')[-1].split('.')[0][0]}{1+int(file.split('_')[-1].split('.')[0][1:])}.{0}"

# print( f"{'_'.join(file.split('_')[:-1])}_{file.split('_')[-1].split('.')[0]}.{1+int(file.split('_')[-1].split('.')[1])}" )
# print( f"{'_'.join(file.split('_')[:-1])}_{file.split('_')[-1].split('.')[0][0]}{1+int(file.split('_')[-1].split('.')[0][1:])}.{0}" )

# file2="WF_hi_v1.3"


# file3="WF_hi_v4.5.h5"
# print( file2.split('.h5')[0] )
# print( len(file2.split('.h5')))
# print( file3.split('.h5')[0] )
# print( len(file3.split('.h5')))


# file4="WF"
# if len( file4.split('_v') )==1: print( file4 + '_v0.0')
# if len( file2.split('_v') )==1: print( file2 + '_v0.0')
# if len( file3.split('_v') )==1: print( file3 + '_v0.0')

# file5=""
# if len( file4.split('_v') )==1: file5= file4.split('.h5')[0] + '_v0.0'
# print(f"5: {file5}")


dict=[]
with open('dict.txt', 'r') as f:
    #print(f.readline())
    #if f.readline(): dict.append( f.readline() )
    # line = f.readline()
    # if  line or not line.isspace(): dict.append( line.lower() )
    for line in f:
        print(line.rstrip())
        dict.append(line.split("\n")[0])
cnt= len(dict)

print(dict)
print(cnt)