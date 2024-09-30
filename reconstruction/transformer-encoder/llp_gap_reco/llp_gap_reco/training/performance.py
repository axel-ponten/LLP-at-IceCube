import numpy as np
import matplotlib.pyplot as plt
import torch

def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def angular_difference_from_points(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    # calculate line 1
    dx1, dy1, dz1 = calculate_line(x1, y1, z1, x2, y2, z2)
    # calculate line 2
    dx2, dy2, dz2 = calculate_line(x3, y3, z3, x4, y4, z4)
    # find angles
    phi1, theta1 = find_angles(dx1, dy1, dz1)
    phi2, theta2 = find_angles(dx2, dy2, dz2)
    # calculate angular difference
    return angular_difference(phi1, theta1, phi2, theta2)

def calculate_line(x1, y1, z1, x2, y2, z2):
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return dx, dy, dz

# Find angles phi and theta
def find_angles(dx, dy, dz):
    phi = np.arctan2(dy, dx)
    theta = np.arctan2(np.sqrt(dx**2 + dy**2), dz)
    return phi, theta

def angular_difference(phi1, eta1, phi2, eta2):
    dphi = np.abs(phi1 - phi2)
    if dphi > np.pi:
        dphi = 2*np.pi - dphi
    deta = np.abs(eta1 - eta2)
    return np.sqrt(dphi**2 + deta**2)

def unnormalize_hits(hits, normalization_args):
    """ Normalize the data. x = (x-offset)*scale"""
    # for each feature type ("log_charges", "position", "abs_time", etc.)
    if normalization_args["position"]["scale"] != 1.0:
        hits /= normalization_args["position"]["scale"]
    if normalization_args["position"]["offset"] != 0.0:
        hits += normalization_args["position"]["offset"]
    return hits

def plot_event(data, prediction, label = None, normalization_args = None, ax = None, title = None):
    # List of 3D positions
    hits = data.squeeze()[:,:3].cpu().numpy()
    charges = data.squeeze()[:,3].cpu().numpy().tolist()
    
    # unnormalize hits for plotting
    if normalization_args is not None:
        hits = unnormalize_hits(hits, normalization_args)
    
    x = [pos[0] for pos in hits]
    y = [pos[1] for pos in hits]
    z = [pos[2] for pos in hits]
    scale = 2
    s = [scale*(10**charge-1) for charge in charges] # total charge from log(1+charge)

    # label
    if label is not None:
        label = label.cpu().numpy().tolist()
        prod_x = label[0]
        prod_y = label[1]
        prod_z = label[2]
        decay_x = label[3]
        decay_y = label[4]
        decay_z = label[5]

    # prediction
    if type(prediction) == torch.Tensor:
        prediction = prediction.cpu().numpy().tolist()
    else:
        prediction = prediction.tolist()
    pred_prod_x = prediction[0]
    pred_prod_y = prediction[1]
    pred_prod_z = prediction[2]
    pred_decay_x = prediction[3]
    pred_decay_y = prediction[4]
    pred_decay_z = prediction[5]

    # Create a 3D plot
    if ax is None:
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')

    # plot hits
    ax.scatter(x, y, z, s=s)
    
    # plot label and draw line between
    if label is not None:
        ax.scatter(prod_x, prod_y, prod_z, color='red')
        ax.scatter(decay_x, decay_y, decay_z, color='blue')
        label_length = distance(prod_x, prod_y, prod_z, decay_x, decay_y, decay_z)
        ax.plot([prod_x, decay_x], [prod_y, decay_y], [prod_z, decay_z], color='red', label='label: {:.2f} m'.format(label_length))
    
    # plot prediction and draw line between
    ax.scatter(pred_prod_x, pred_prod_y, pred_prod_z, color='red')
    ax.scatter(pred_decay_x, pred_decay_y, pred_decay_z, color='blue')
    pred_length = distance(pred_prod_x, pred_prod_y, pred_prod_z, pred_decay_x, pred_decay_y, pred_decay_z)
    ax.plot([pred_prod_x, pred_decay_x], [pred_prod_y, pred_decay_y], [pred_prod_z, pred_decay_z], color='black', label='pred: {:.2f} m'.format(pred_length))

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Red: Production vertex, Blue: Decay vertex')

    # Set all axes limits
    xmin = -600
    xmax = 600
    ymin = -600
    ymax = 600
    zmin = -600
    zmax = 600
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.legend()
    
    # default view some degree from line azimuth
    azi_pred = np.arctan2(pred_decay_y - pred_prod_y, pred_decay_x - pred_prod_x) * 180 / np.pi
    ax.view_init(elev=20, azim=azi_pred + 135.0)
    
    return


def plot_2x2_fn(data_list, prediction_list, label_list = None, normalization_args = None):
    if len(data_list) != len(prediction_list):
        raise ValueError("data, label and prediction must have the same length")
    if len(data_list) != 4:
        raise ValueError("data must have 4 events")
    
    # Create a 3D plot
    fig = plt.figure(figsize=(14,14))  # Set the figure size to 10x8 inches
    plt.title('Red: Production vertex, Blue: Decay vertex')
    plt.subplots_adjust(hspace=0., wspace=0., left=0., right=0.9, top=0.9, bottom=0.)
    # plt.tight_layout()
    # add subplots
    if label_list is None:
        label_list = [None]*4
    for i, (data, label, prediction) in enumerate(zip(data_list, label_list, prediction_list)):
        ax = fig.add_subplot(221+i, projection='3d')
        
        # List of 3D positions
        hits = data.squeeze()[:,:3].cpu().numpy()
        charges = data.squeeze()[:,3].cpu().numpy().tolist()
        
        # unnormalize hits for plotting
        if normalization_args is not None:
            hits = unnormalize_hits(hits, normalization_args)
        
        x = [pos[0] for pos in hits]
        y = [pos[1] for pos in hits]
        z = [pos[2] for pos in hits]
        scale = 1
        s = [scale*(10**charge-1) for charge in charges] # total charge from log(1+charge)
            
        # label
        if label is not None:
            label = label.cpu().numpy().tolist()
            prod_x = label[0]
            prod_y = label[1]
            prod_z = label[2]
            decay_x = label[3]
            decay_y = label[4]
            decay_z = label[5]

        # prediction
        if type(prediction) == torch.Tensor:
            prediction = prediction.cpu().numpy().tolist()
        else:
            prediction = prediction.tolist()
        pred_prod_x = prediction[0]
        pred_prod_y = prediction[1]
        pred_prod_z = prediction[2]
        pred_decay_x = prediction[3]
        pred_decay_y = prediction[4]
        pred_decay_z = prediction[5]

        # plot hits
        ax.scatter(x, y, z, s=s)
        
        # plot label and draw line between
        if label is not None:
            ax.scatter(prod_x, prod_y, prod_z, color='red')
            ax.scatter(decay_x, decay_y, decay_z, color='blue')
            label_length = distance(prod_x, prod_y, prod_z, decay_x, decay_y, decay_z)
            ax.plot([prod_x, decay_x], [prod_y, decay_y], [prod_z, decay_z], color='red', label='label: {:.2f} m'.format(label_length))
        
        # plot prediction and draw line between
        ax.scatter(pred_prod_x, pred_prod_y, pred_prod_z, color='red')
        ax.scatter(pred_decay_x, pred_decay_y, pred_decay_z, color='blue')
        pred_length = distance(pred_prod_x, pred_prod_y, pred_prod_z, pred_decay_x, pred_decay_y, pred_decay_z)
        ax.plot([pred_prod_x, pred_decay_x], [pred_prod_y, pred_decay_y], [pred_prod_z, pred_decay_z], color='black', label='pred: {:.2f} m'.format(pred_length))

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set all axes limits
        xmin = -600
        xmax = 600
        ymin = -600
        ymax = 600
        zmin = -600
        zmax = 600
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])

        ax.legend()


def cnf_corner_plot():
    pass