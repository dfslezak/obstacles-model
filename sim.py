import numpy as np

def oneSim(p=0.5,NSIMS = 75000,MAX_DIST = 11):
    MAX_STEPS = 2000
    #NSIMS = 75000
    rand_dir = np.random.randint(1,5,(NSIMS, MAX_STEPS))
    ph_x = np.zeros((NSIMS,MAX_STEPS))
    ph_x[rand_dir==1] = 1.0
    ph_x[rand_dir==2] = -1.0
    ph_y = np.zeros((NSIMS,MAX_STEPS))
    ph_y[rand_dir==3] = 1.0
    ph_y[rand_dir==4] = -1.0

    ph_posicion_x, ph_posicion_y = np.array(np.meshgrid(np.arange(MAX_STEPS,0,-1), np.arange(NSIMS))).astype(float)

    rand_p = np.random.rand(NSIMS, MAX_STEPS)
    ph_p = (rand_p>p).astype(float)

    x_cum = np.cumsum(np.multiply(ph_x,ph_p),axis=1)
    y_cum = np.cumsum(np.multiply(ph_y,ph_p),axis=1)
    distancia_cuadrada = np.multiply(x_cum,x_cum) + np.multiply(y_cum,y_cum)

    result = np.zeros((MAX_DIST+1,NSIMS))
    for i in range(1,MAX_DIST+1):
        distancia_umbral = distancia_cuadrada>(i*i)
        posicion = np.multiply(ph_posicion_x,distancia_umbral)
        pasos = np.subtract(MAX_STEPS+1.0,np.max(posicion,axis=1))
        result[i] = pasos
    return result
