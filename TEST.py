def cal_box( points:np.array, dis_x:np.array ,dis_y:np.array,dest_width:np.array, 
        dest_height:np.array,width_ratio:np.array,height_ratio:np.array):
    lens = points.shape[0]
    box = []
    for iii in range(lens): 
        xx,yy = points[iii]
        offset_x = dis_x[yy][xx]
        offset_y = dis_y[yy][xx]
        box.append([xx+offset_x, yy+offset_y])
        box = np.array(box).astype(np.int16)
        box[:, 0] = box[:, 0]* width_ratio
        box[:, 0][box[:, 0]>dest_width] = dest_width
        box[:, 1] = box[:, 1]* height_ratio
        box[:, 1][box[:, 1]>dest_height] = dest_height
    return box
def cal_box( points:np.array, dis_x:np.array ,dis_y:np.array,dest_width:np.array, 
dest_height:np.array,width_ratio:np.array,height_ratio:np.array):
lens = points.shape[0]
box = []
for iii in range(lens):  
xx,yy = points[iii]
offset_x = dis_x[yy][xx]
offset_y = dis_y[yy][xx]
box.append([xx+offset_x, yy+offset_y])
box = np.array(box).astype(np.int16)
box[:, 0] = box[:, 0]* width_ratio
box[:, 0][box[:, 0]>dest_width] = dest_width
box[:, 1] = box[:, 1]* height_ratio
box[:, 1][box[:, 1]>dest_height] = dest_height
return box