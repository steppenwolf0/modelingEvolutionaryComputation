import numpy as np

from scipy.optimize import minimize
from math import exp,fabs

def testingModel(xtrain):
	Dimension=20
	cost=0.0
	
	x = np.zeros(Dimension)

	for i in range (0,Dimension):
		x[i] = (2.0 * xtrain[i] - 1.0)/1.0

	
	v0=np.zeros(5)
	v1=np.zeros(5)
	v2=np.zeros(5)
	v3=np.zeros(5)
	v4=np.zeros(5)

	v0[0]=3.768626875
	v0[1]=3.46085785
	v0[2]=3.446518574
	v0[3]=3.518292143
	v0[4]=3.452836989
				
	v1[0]=3.742900853
	v1[1]=3.457884586	
	v1[2]=3.417206836
	v1[3]=3.336099096
	v1[4]=3.505254216
				
	v2[0]=3.740509442
	v2[1]=3.503851115
	v2[2]=3.398431563
	v2[3]=3.451343139
	v2[4]=3.479362452
				
	v3[0]=7.933538493
	v3[1]=7.672636459
	v3[2]=7.545667958
	v3[3]=7.578043433
	v3[4]=7.686156809
				
	v4[0]=4.337866717
	v4[1]=4.80250431
	v4[2]=4.392954553
	v4[3]=4.735453756
	v4[4]=4.77950282

	v_0=v0[0]
	v_1=v1[0]
	v_2=v2[0]
	v_3=v3[0]
	v_4=v4[0]
	
	u=(v_0*x[15]+v_1*x[16]+v_2*x[17]+v_3*x[18]+v_4*x[19])

	diffT=0.25
	t=0.0
	vars=int(183/diffT)
	for j in range(0,vars):
	
		u=(v_0*x[15]+v_1*x[16]+v_2*x[17]+v_3*x[18]+v_4*x[19])
			
		dv0=-x[0]*u+x[5]*exp(-x[6]*t)
		v_0=v_0+dv0*diffT
		dv1=-x[1]*u+x[7]*exp(-x[8]*t)
		v_1=v_1+dv1*diffT
		dv2=-x[2]*u+x[9]*exp(-x[10]*t)
		v_2=v_2+dv2*diffT
		dv3=-x[3]*u+x[11]*exp(-x[12]*t)
		v_3=v_3+dv3*diffT
		dv4=-x[4]*u+x[13]*exp(-x[14]*t)
		v_4=v_4+dv4*diffT
		
		t+=diffT
		
		if(t==7):
			cost+=fabs(v_0-v0[1])
			cost+=fabs(v_1-v1[1])
			cost+=fabs(v_2-v2[1])
			cost+=fabs(v_3-v3[1])
			cost+=fabs(v_4-v4[1])
		if (t==42):
			cost+=fabs(v_0-v0[2])
			cost+=fabs(v_1-v1[2])
			cost+=fabs(v_2-v2[2])
			cost+=fabs(v_3-v3[2])
			cost+=fabs(v_4-v4[2])
		if (t==98):
			cost+=fabs(v_0-v0[3])
			cost+=fabs(v_1-v1[3])
			cost+=fabs(v_2-v2[3])
			cost+=fabs(v_3-v3[3])
			cost+=fabs(v_4-v4[3])
		if (t==182):
			cost+=fabs(v_0-v0[4])
			cost+=fabs(v_1-v1[4])
			cost+=fabs(v_2-v2[4])
			cost+=fabs(v_3-v3[4])
			cost+=fabs(v_4-v4[4])

	v0	[0]=3.41314195
	v0	[1]=3.707415382
	v0	[2]=3.557259467
	v0	[3]=3.481412223
	v0	[4]=3.453528466
					
	v1	[0]=3.382489987
	v1	[1]=3.436153008
	v1	[2]=3.488596267
	v1	[3]=3.499065554
	v1	[4]=3.419799543
					
	v2	[0]=3.345835179
	v2	[1]=3.388451251
	v2	[2]=3.368207656
	v2	[3]=3.368545656
	v2	[4]=3.413106589
					
	v3	[0]=7.563551498
	v3	[1]=7.570545446
	v3	[2]=7.581871317
	v3	[3]=7.577028926
	v3	[4]=7.577353775
					
	v4	[0]=5.449667976
	v4	[1]=5.391162525
	v4	[2]=5.443780733
	v4	[3]=5.613788687
	v4	[4]=5.530816341
	
	v_0=v0[0]
	v_1=v1[0]
	v_2=v2[0]
	v_3=v3[0]
	v_4=v4[0]
	
	u=(v_0*x[15]+v_1*x[16]+v_2*x[17]+v_3*x[18]+v_4*x[19])

	diffT=0.25
	t=0.0
	vars=int(183/diffT)
	for j in range(0,vars):
	
		u=(v_0*x[15]+v_1*x[16]+v_2*x[17]+v_3*x[18]+v_4*x[19])
			
		dv0=-x[0]*u+x[5]*exp(-x[6]*t)
		v_0=v_0+dv0*diffT
		dv1=-x[1]*u+x[7]*exp(-x[8]*t)
		v_1=v_1+dv1*diffT
		dv2=-x[2]*u+x[9]*exp(-x[10]*t)
		v_2=v_2+dv2*diffT
		dv3=-x[3]*u+x[11]*exp(-x[12]*t)
		v_3=v_3+dv3*diffT
		dv4=-x[4]*u+x[13]*exp(-x[14]*t)
		v_4=v_4+dv4*diffT
		
		t+=diffT
		
		if(t==7):
			cost+=fabs(v_0-v0[1])
			cost+=fabs(v_1-v1[1])
			cost+=fabs(v_2-v2[1])
			cost+=fabs(v_3-v3[1])
			cost+=fabs(v_4-v4[1])
		if (t==42):
			cost+=fabs(v_0-v0[2])
			cost+=fabs(v_1-v1[2])
			cost+=fabs(v_2-v2[2])
			cost+=fabs(v_3-v3[2])
			cost+=fabs(v_4-v4[2])
		if (t==98):
			cost+=fabs(v_0-v0[3])
			cost+=fabs(v_1-v1[3])
			cost+=fabs(v_2-v2[3])
			cost+=fabs(v_3-v3[3])
			cost+=fabs(v_4-v4[3])
		if (t==182):
			cost+=fabs(v_0-v0[4])
			cost+=fabs(v_1-v1[4])
			cost+=fabs(v_2-v2[4])
			cost+=fabs(v_3-v3[4])
			cost+=fabs(v_4-v4[4])

	return cost


x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5])

res = minimize(testingModel, x0, method='nelder-mead',options={'xatol': 1e-8, 'disp': True, 'maxiter':30000 })

print(testingModel(res.x))

print(res.x)