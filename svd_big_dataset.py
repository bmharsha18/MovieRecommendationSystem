import csv
import numpy as np

def SVD(M):
    #Calculation part of U in svd
    prd=np.dot(M,M.T)
    #Calculate eigen values and eigen vectors of M*M.T
    eigenvalue,eigenvec=np.linalg.eig(prd)
    #sort the eigen values in decreasing order
    sortindex=eigenvalue.argsort()[::-1]
    eigenvalue=eigenvalue[sortindex]    

    #To calculate sigma
    sigma=np.sqrt(abs(eigenvalue))
    sigma=np.around(sigma,decimals=2)
    #Calculate the sum of the values in sigma matrix    
    totalsigma=np.sum(sigma,dtype=float)
  
    #To calculate the value of the dimension based on the input percentage of how much percent data should be retained
    percent=int(input('Enter Percentage of variance of original data to be retained -'))
    data=(percent/100)*totalsigma
    sumsigma=0.0

    #To calculate required dimensions
    dim=0
    while(sumsigma<=data):
        sumsigma+=sigma[dim]
        dim+=1
    print('We need', dim, 'components to preserve ', (sumsigma/totalsigma)*100,' variance of data')
    #reduce sigma to 0 to dim
    sigma=sigma[0:dim]
    #reduce U to 0 to dim
    U=eigenvec[:,sortindex]
    U=U[:,0:dim]
    #if U contains imaginary numbers, extract only real part of it
    U=np.real(U)
    #scale upto 2 decimals for easier understanding
    U=np.around(U,decimals=2)
    
    #To Calculate V in the similar manner for U but here compute for M.T*M
    prd=np.dot(M.T,M)
    eigenvalue,eigenvec=np.linalg.eig(prd)
    sortindex=eigenvalue.argsort()[::-1]
    V=eigenvec[:,sortindex]
    V=V[:,0:dim]
    V=np.real(V)
    V=np.around(V,decimals=2) 
    
    return U,sigma,V
    
def query(q,V):
    #to calculate strength/prediction rate for each user for each movie
    prd=np.dot(q,V)
    Vt=np.transpose(V)
    other=np.dot(prd,Vt)
    return other


#To Prepare list of movies - for recommending
fileh=open('u.item','r')
reader = csv.reader(fileh, delimiter='|')
movienames=list()
# The list of all the movies with movieid-1 as list index
for row in reader:
	movienames.append(row[1])

num_users=943
num_movies=1682

#To Prepare matrix M 
fp2=open('u.data','r')
reader = csv.reader(fp2, delimiter='\t')
m=list()
for j in range(num_users):
	m.append([0]*num_movies)
for row in reader:
	m[int(row[0])-1][int(row[1])-1]=float(row[2])
	
M=np.array(m)

U,sigma,V=SVD(M)

#To predict movies for a user.
uid=int(input("Enter userid"))    
q=m[uid-1]
predict=query(q,V)

#Sorting the user_rating row based on index
idx=predict.argsort()[::-1]
predicted=predict[idx]

#To display 10 movies, can change it by taking input from user
nm=10
i=0
j=0
mr=list()
print("\n\nMovie Recommendation\n")
while(i<nm):
    if(m[uid-1][idx[j]]==0):
        mr1=list()
        mr1.append(idx[j])
        mr1.append(movienames[idx[j]-1])
        mr.append(mr1)
        i+=1
    j+=1

print("MovieID\tMovieTittle\n")
for i in mr:
    print(i[0],"\t",i[1])


