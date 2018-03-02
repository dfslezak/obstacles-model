import numpy as np

def oneSimMaze(obstacles=0.5,NSIMS = 75000,MAX_DIST = 11,maze=None):
    MAX_STEPS = 2000
    #   NSIMS = 75000
    rand_dir = np.random.randint(1,5,(NSIMS, MAX_STEPS))
    ph_x = np.zeros((NSIMS,MAX_STEPS))
    ph_x[rand_dir==1] = 1.0
    ph_x[rand_dir==2] = -1.0
    ph_y = np.zeros((NSIMS,MAX_STEPS))
    ph_y[rand_dir==3] = 1.0
    ph_y[rand_dir==4] = -1.0

    ph_posicion_x, ph_posicion_y = np.array(np.meshgrid(np.arange(MAX_STEPS,0,-1), np.arange(NSIMS))).astype(float)

    if maze is None:
        rand_p = np.random.rand(2*MAX_STEPS,2*MAX_STEPS)
        maze = (rand_p>obstacles)

    x_cum = np.zeros((NSIMS,MAX_STEPS+1))
    y_cum = np.zeros((NSIMS,MAX_STEPS+1))

    check_maze_without_obs = lambda x,y: maze[x,y]
    vfunc = np.vectorize(check_maze_without_obs)

    for i in range(1,MAX_STEPS+1):
        cambio_maze = vfunc(x_cum[:,i-1].astype(int)+ph_x[:,i-1].astype(int)+MAX_DIST,
                            y_cum[:,i-1].astype(int)+ph_y[:,i-1].astype(int)+MAX_DIST)
        x_cum[cambio_maze,i] = x_cum[cambio_maze,i-1]+ph_x[cambio_maze,i-1]
        y_cum[cambio_maze,i] = y_cum[cambio_maze,i-1]+ph_y[cambio_maze,i-1]
        x_cum[~cambio_maze,i] = x_cum[~cambio_maze,i-1]
        y_cum[~cambio_maze,i] = y_cum[~cambio_maze,i-1]

    distancia_cuadrada = np.multiply(x_cum[:,1:],x_cum[:,1:]) + np.multiply(y_cum[:,1:],y_cum[:,1:])

    result = np.zeros((MAX_DIST+1,NSIMS))
    for i in range(1,MAX_DIST+1):
        distancia_umbral = distancia_cuadrada>(i*i)
        posicion = np.multiply(ph_posicion_x,distancia_umbral)
        pasos = np.subtract(MAX_STEPS+1.0,np.max(posicion,axis=1))
        result[i] = pasos
    return result

def oneSimMazeFull(obstacles=0.5,total_runs = 75000,MAX_DIST = 11):
    SIMS_PER_MAZE = 1000
    pasos_obs = oneSimMaze(obstacles=obstacles,NSIMS=SIMS_PER_MAZE,MAX_DIST=MAX_DIST)

    for i in range((total_runs/SIMS_PER_MAZE)-1):
        pasos_temp = oneSimMaze(obstacles=obstacles,NSIMS=SIMS_PER_MAZE,MAX_DIST=MAX_DIST)
        pasos_obs = np.concatenate([pasos_obs,pasos_temp],axis=1)

    return pasos_obs
