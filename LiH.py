import sys
import vqe_methods_add_by_one_Harper_truncation
import numpy
import pickle
import numpy as np


min_dist = float(sys.argv[1])
max_dist = float(sys.argv[2])
num_step = int(sys.argv[3])
id_step = int(sys.argv[4])

dist_list = list(np.linspace(min_dist, max_dist, num_step))
dist_list = list(map(lambda x: round(x, 5), dist_list))


dist=dist_list[id_step]

print(f'The list of distances to compute is: {dist_list}')
print(f'I am the distance: {dist}')


PoolOfHope=['XYYZIIZIZY', 'XYYYIZZZII', 'YYIZZZIZXY', 'XXZXZIIIYI', 'XYZYIZZIYI', 'XXXZIIZZZY', 'XXIIYXZZII', 'XYXZXXXYZY', 'XXIYIIXYZY', 'IIZIZZYYXY', 'ZZXZXXIIZY', 'YZZZXYZZZY', 'YXZZIZYYII', 'IXIZXXZZYI']
Resultat=[]
    
    
    
    
geometry = "Li 0 0 0; H 0 0 {}".format(dist)
print(geometry)
vqe_methods_add_by_one_Harper_truncation.adapt_vqe(geometry,
	                  adapt_thresh    = 1e-8,                        #gradient threshold
                      adapt_maxiter   = 400,                       #maximum number of ops                   
                      Pool            = PoolOfHope,
                      Resultat        = Resultat,
                      bond_legth      = dist
                      ) 
        
with open('Bond_length_dependence.LiH_dissociation_curve_pickle_min_pool_{}'.format(dist) , 'wb') as handle:
    pickle.dump(Resultat, handle, protocol=pickle.HIGHEST_PROTOCOL)                        

