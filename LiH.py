import sys
import vqe_methods_add_by_one_Harper_truncation
import numpy
import pickle


dist=1.0#sys.argv[1]


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

