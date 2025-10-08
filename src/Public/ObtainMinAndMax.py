import numpy as np

def obtainMinAndMax(problem, m):
    prefix = problem.split("_")[0]
    
    if prefix in ['DTLZ1']:
        zmin = np.zeros(m)
        zmax = np.ones(m)*(0.5)
    elif prefix in ['DTLZ2', 'DTLZ3', 'DTLZ4', 'IMOP1', 'IMOP2', 'IMOP4', 'IMOP6', 'IMOP7']:
        zmin = np.zeros(m)
        zmax = np.ones(m)
    elif prefix in ['DTLZ5']:
        zmin = np.zeros(m)
        if m == 2:
            zmax = np.array([1, 1])
        elif m == 3:
            zmax = np.array([7.071068e-01, 7.071068e-01, 1])
        elif m == 5:
            zmax = np.array([3.120704e+00, 2.556749e+00, 3.326446e+00, 3.412240e+00, 3.490394e+00])
        elif m == 8:
            zmax = np.array([2.982901e+00, 2.985621e+00, 3.060304e+00, 3.160221e+00, 3.241811e+00, 3.325741e+00, 3.412237e+00, 3.463355e+00])
        elif m == 10:
            zmax = np.array([2.841839e+00, 2.841919e+00, 2.916733e+00, 3.003657e+00, 3.079689e+00, 3.161682e+00, 3.243226e+00, 3.326684e+00, 3.412248e+00, 3.492054e+00])
    elif prefix in ['DTLZ6']:
        zmin = np.zeros(m)
        if m == 3:
            zmax = np.array([7.071068e-01, 7.071068e-01, 1])
        elif m == 5:
            zmax = np.array([8.032197e+00, 8.594481e+00, 1.061013e+01, 1.097197e+01, 1.068246e+01])
        elif m == 8:
            zmax = np.array([1.071609e+01, 1.074065e+01, 1.077817e+01, 1.086285e+01, 1.089971e+01, 1.093316e+01, 1.097188e+01, 1.067303e+01])
        elif m == 10:
            zmax = np.array([1.076331e+01, 1.072612e+01, 1.079803e+01, 1.082174e+01, 1.085286e+01, 1.088450e+01, 1.091439e+01, 1.094281e+01, 1.097197e+01, 1.062296e+01])
    elif prefix in ['DTLZ7']:
        zmin = np.zeros(m-1)
        zmax = np.append(np.ones(m-1)*(0.859401), 2*m)
        if m == 2:
            zmin = np.append(zmin, 2)
        elif m == 3:
            zmin = np.append(zmin, 2.614009)
        elif m == 5:
            zmin = np.append(zmin, 3.228017)
        elif m == 8:
            zmin = np.append(zmin, 4.149031)
        elif m == 10:
            zmin = np.append(zmin, 4.763039)
    elif prefix in ['DTLZ1_MINUS']:
        zmin = np.ones(m)*(-5.511507e+02)
        zmax = np.zeros(m)
    elif prefix in ['DTLZ2_MINUS', 'DTLZ4_MINUS']:
        zmin = np.ones(m)*(-3.5)
        zmax = np.zeros(m)
    elif prefix in ['DTLZ3_MINUS']:
        zmin = np.ones(m)*(-2.203603e+03)
        zmax = np.zeros(m)
    elif prefix in ['DTLZ5_MINUS']:
        zmax = np.zeros(m)
        if m == 3:
            zmin = np.array([-3.412248e+00, -3.412248e+00, -3.500000e+00])
        elif m == 5:
            zmin = np.array([-3.243288e+00, -3.243288e+00, -3.326696e+00, -3.412248e+00, -3.500000e+00])
        elif m == 8:
            zmin = np.array([-3.005405e+00, -3.005405e+00, -3.082695e+00, -3.161972e+00, -3.243288e+00, -3.326696e+00, -3.412248e+00, -3.500000e+00])
        elif m == 10:
            zmin = np.array([-2.856591e+00, -2.856591e+00, -2.930054e+00, -3.005405e+00, -3.082695e+00, -3.161972e+00, -3.243288e+00, -3.326696e+00, -3.412248e+00, -3.500000e+00])
    elif prefix in ['DTLZ6_MINUS']:
        zmax = np.zeros(m)
        if m == 3:
            zmin = np.array([-1.097197e+01, -1.097197e+01, -1.100000e+01])
        elif m == 5:
            zmin = np.array([-1.091613e+01, -1.091613e+01, -1.094402e+01, -1.097197e+01, -1.100000e+01])
        elif m == 8:
            zmin = np.array([-1.083287e+01, -1.083288e+01, -1.086055e+01, -1.088831e+01, -1.091613e+01, -1.094402e+01, -1.097197e+01, -1.100000e+01])
        elif m == 10:
            zmin = np.array([-1.077771e+01, -1.077770e+01, -1.080527e+01, -1.083289e+01, -1.086057e+01, -1.088832e+01, -1.091613e+01, -1.094402e+01, -1.097197e+01, -1.100000e+01])
    elif prefix in ['DTLZ7_MINUS']:
        zmin = np.ones(m-1)*(-1)
        zmax = np.zeros(m-1)
        if m == 3:
            zmin = np.append(zmin, -33)
            zmax = np.append(zmax, -31)
        elif m == 5:
            zmin = np.append(zmin, -55)
            zmax = np.append(zmax, -51)
        elif m == 8:
            zmin = np.append(zmin, -88)
            zmax = np.append(zmax, -81)
        elif m == 10:
            zmin = np.append(zmin, -110)
            zmax = np.append(zmax, -101)
    elif prefix in ['WFG1', 'WFG2', 'WFG4', 'WFG5', 'WFG6', 'WFG7', 'WFG8', 'WFG9']:
        zmin = np.zeros(m)
        zmax = np.arange(2.0, 2*m+1, 2)
    elif prefix in ['WFG3']:
        zmin = np.zeros(m)
        if m == 2:
            zmax = np.array([2.998313e+00, 3.281663e+00])
        elif m == 3:
            zmax = np.array([2.988887e+00, 3.281663e+00, 6.081809e+00])
        elif m == 5:
            zmax = np.array([2.998313e+00, 4.996260e+00, 6.996529e+00, 8.237908e+00, 1.013179e+01])
        elif m == 8:
            zmax = np.array([2.999996e+00, 4.999992e+00, 6.999991e+00, 8.999991e+00, 1.099999e+01, 1.299999e+01, 1.484754e+01, 1.660957e+01])
        elif m == 10:
            zmax = np.array([3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00, 1.100000e+01, 1.300000e+01, 1.500000e+01, 1.700000e+01, 1.895306e+01, 2.064856e+01])
    elif prefix in ['WFG1_MINUS', 'WFG2_MINUS', 'WFG3_MINUS', 'WFG4_MINUS', 'WFG5_MINUS', 'WFG6_MINUS', 'WFG7_MINUS', 'WFG8_MINUS', 'WFG9_MINUS']:
        zmin = np.arange(-3.0, -2*m-2, -2)
        zmax = np.ones(m)*(-1)
    elif prefix in ['IMOP3']:
        zmin = np.array([-1.025384e-01, 0.000000e+00])
        zmax = np.array([1.200000e+00, 9.050877e-01])
    elif prefix in ['IMOP5']:
        zmin = np.array([-5.000000e-01, -5.000000e-01, -2.071062e-01])
        zmax = np.array([5.000000e-01, 5.000000e-01, 1.207106e+00])
    elif prefix in ['IMOP8']:
        zmin = np.array([0.000000e+00, 0.000000e+00, -8.958892e-01])
        zmax = np.array([9.742647e-01, 9.742647e-01, 3.000000e+00])
    elif prefix in ['VNT1']:
        zmin = np.array([0.000000e+00, 1.000000e+00, 2.000000e+00])
        zmax = np.array([4.000000e+00, 5.000000e+00, 4.000000e+00])
    elif prefix in ['VNT2']:
        zmin = np.array([3.000000e+00, -1.700000e+01, -1.300000e+01])
        zmax = np.array([4.236328e+00, -1.647913e+01, -1.205311e+01])
    elif prefix in ['VNT3']:
        zmin = np.array([0.000000e+00, 1.500000e+01, -1.000000e-01])
        zmax = np.array([8.196103e+00, 1.703704e+01, 1.761271e-01])
    elif 'ZCAT' in prefix:
        zmin = np.zeros(m)
        zmax = np.arange(1, m+1)**2
    elif 'ZDT' in prefix:
        zmin = np.zeros(m)
        zmax = np.ones(m)        
    return zmin, zmax
