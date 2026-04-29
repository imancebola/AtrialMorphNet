import torch
import plotly.graph_objects as go


def plot_pc(pc, title=""):
    pc = pc.cpu().detach().numpy()
    fig = go.Figure(data=[go.Scatter3d(x=pc[:,0], y=pc[:,1], z=pc[:,2],
                                       mode='markers', marker=dict(size=2, color=pc[:,2], colorscale='Viridis'))])
    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    fig.show()