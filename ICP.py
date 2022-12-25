import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def GetDistance(a, b):
  return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def ICP_Init():
  csv = pd.read_csv('point_cloud.csv')
  # Do I need to handle -inf, inf?
  positions = []
  for position in csv['position']:
    positions.append(list(map(float, position.split('#'))))
  clouds = []
  for points in csv['point_cloud']:
    string_points = list(points.split('#'))
    cloud = []
    for point in string_points:
      cloud.append(list(map(float, point.split(','))))
    clouds.append(cloud)
  abs_pos = np.zeros_like(positions)
  # Need to check abs_pos
  for i in range(len(positions)):
    abs_pos[i][0] = positions[i][0] - positions[0][0]
    abs_pos[i][1] = positions[i][1] - positions[0][1]
    abs_pos[i][2] = positions[i][2] - positions[0][2]
  return abs_pos, clouds

def ICP(A, B, max_iter):
  converge = False
  count = max_iter
  X = A
  P = B
  match = [[] for point in P]
  transition = np.zeros(2) # [x, y]
  rotation = np.array([[1.0, 0.0], [0.0, 1.0]])
  total_angle = 0
  while not converge:
    # Find closest point
    for i in range(len(P)):
      min_idx = i
      min_dist = np.inf
      dist = 0
      for j in range(len(X)):
        dist = GetDistance(P[i], X[j])
        if dist < min_dist:
          min_idx = j
          min_dist = dist
      if min_dist == np.inf:
        min_dist = -1
      match[i] = [min_idx, min_dist]
    # Get algined points
    for i in range(len(match)):
      min_idx = i
      min_dist = match[i][1]
      match[i][1] = -1
      if min_dist != -1:
        for j in range(len(match)):
          if i != j and match[i][0] == match[j][0]:
            if match[j][1] != -1 and match[j][1] < min_dist:
              min_idx = j
              min_dist = match[j][1]
              match[j][1] = -1
            else:
              match[j][1] = -1
        match[min_idx][1] = min_dist
    # solve transition, rotation
    x_sum = np.zeros(2)
    p_sum = np.zeros(2)
    cov = np.zeros((2,2))
    corr_count = 0
    for i in range(len(match)):
      if match[i][1] != -1:
        x_sum += np.array(X[match[i][0]])
        p_sum += np.array(P[i])
        curr_cov = np.outer(np.array(P[i]).transpose(), np.array(X[match[i][0]]))
        cov += curr_cov
        corr_count += 1
    x_mean = x_sum / corr_count
    p_mean = p_sum / corr_count
    U, S, V_t = np.linalg.svd(cov)
    R = np.matmul(V_t.transpose(), U.transpose())
    angle = math.atan2(R[1][0], R[0][0]) * 180 / np.pi
    T = np.array(x_mean - p_mean)
    # apply to P
    P = np.array(np.matmul(R, np.array(P).transpose())).transpose() + T
    transition += T
    rotation = np.matmul(R, rotation)
    total_angle += angle
    count -= 1
    if count == 0: # How to check convergence
      converge = True
  return transition, rotation, angle

abs_pos, clouds = ICP_Init()
''' For saving original plots
for i in range(len(clouds)):
  RGB = ((255-20*i)/255, 0, 20*i/255)
  for j in range(len(clouds[i])):
    plt.plot(clouds[i][j][0], clouds[i][j][1], '.', color=RGB)
  title = 'point_cloud_{}.svg'.format(i)
  plt.savefig(title)
  plt.clf()
'''
transition = [[] for _ in clouds]
rotation = [[] for _ in clouds]
angle = [[] for _ in clouds]
#abs_pos = abs_pos[6:10]
#clouds = clouds[6:10]
for i in range(len(clouds)):
  if i != 0:
    transition[i], rotation[i], angle[i] = ICP(clouds[i-1], clouds[i], 60)
    transition[i] = np.array(transition[i])
    rotation[i] = np.array(rotation[i])
    angle[i] = np.array(angle[i])
newclouds = [[] for _ in clouds]
for i in range(len(clouds)):
  if i == 0:
    newclouds[i] = clouds[i]
    newclouds[i].append([0, 0]) # estimated position
  else:
    newclouds[i] = clouds[i]
    newclouds[i].append([0, 0])
    for j in range(i):
      newclouds[i] = np.array(np.matmul(rotation[i-j], np.array(newclouds[i]).transpose())).transpose()
      + transition[i-j]
  RGB = ((255-20*i)/255, 0, 20*i/255)
  for j in range(len(clouds[i])):
    if j == len(clouds[i])-1:
      plt.plot(newclouds[i][j][0], newclouds[i][j][1], 'k*', color=RGB, markersize=8)
    else:
      plt.plot(newclouds[i][j][0], newclouds[i][j][1], '.', color=RGB)
  plt.plot(abs_pos[i][0], abs_pos[i][1],'ko', fillstyle='none',markeredgewidth=3, markersize=16)
plt.savefig('ICP.svg')
plt.show()