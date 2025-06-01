import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import torch
from copy import deepcopy
from my import percentage_overlap
def plot_accuracy(df, base_dir, set_name, arch):
    """Plot train and test accuracy over epochs."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Epoch', y='Train_Acc@1', hue='Generation', data=df, color='blue')
    sns.lineplot(x='Epoch', y='Test_Acc@1', hue='Generation', data=df, color='orange', marker='o')
    for epoch in df[df['Mask_Update']]['Epoch'].unique():
        plt.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
    plt.title(f'Train and Test Accuracy Over Epochs {set_name}, {arch}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(['Train Acc@1', 'Test Acc@1', 'Mask Update'])
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, 'accuracy_over_epochs.png'))
    plt.close()

def plot_loss(df, base_dir, set_name, arch):
    """Plot train and test loss over epochs using Plotly."""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for idx, gen in enumerate(df['Generation'].unique()):
        gen_df = df[df['Generation'] == gen]
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(x=gen_df['Epoch'], y=gen_df['Train_Loss'], mode='lines', name=f'Train Loss Gen {gen}', line=dict(color=color, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=gen_df['Epoch'], y=gen_df['Test_Loss'], mode='lines+markers', name=f'Test Loss Gen {gen}', line=dict(color=color, width=2, dash='dash'), marker=dict(size=8)), row=1, col=1)
    for epoch in df[df['Mask_Update']]['Epoch'].unique():
        fig.add_vline(x=epoch, line=dict(color='red', dash='dash', width=1), opacity=0.5)
    fig.update_layout(title=f'Train and Test Loss Over Epochs ({set_name}, {arch})', xaxis_title='Epoch', yaxis_title='Loss', showlegend=True, template='plotly_white', legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='Black', borderwidth=1))
    fig.write_html(os.path.join(base_dir, 'loss_over_epochs.html'))
    fig.write_image(os.path.join(base_dir, 'loss_over_epochs.png'))

def plot_sparsity(mask_history, base_dir, set_name, arch):
    """Plot sparsity across layers using Seaborn."""
    sparsity_data = []
    for gen in mask_history:
        for layer_name, mask in mask_history[gen].items():
            sparsity = 100 * (1 - mask.mean())
            sparsity_data.append({'Generation': gen, 'Layer': layer_name, 'Sparsity': sparsity})
    sparsity_df = pd.DataFrame(sparsity_data)
    plt.figure(figsize=(14, 6))
    sns.barplot(x='Layer', y='Sparsity', hue='Generation', data=sparsity_df)
    plt.title(f'Sparsity Across Layers at Mask Update Points {set_name}, {arch}')
    plt.xlabel('Layer')
    plt.ylabel('Sparsity (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Generation')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'sparsity_across_layers.png'))
    plt.close()

def plot_layer_sparsity(epoch_metrics, cfg, base_dir, set_name, arch):
    """Plot layer-wise sparsity over epochs using Seaborn."""
    selected_layers = ['conv1.weight', 'layer1.0.conv1.weight', 'layer2.0.conv1.weight', 'layer3.0.conv1.weight', 'layer4.0.conv1.weight', 'fc.weight']
    for gen in range(cfg.num_generations):
        layer_sparsity_data = []
        for epoch in range(cfg.epochs):
            for layer in selected_layers:
                if layer in epoch_metrics['layer_sparsity']:
                    sparsity = epoch_metrics['layer_sparsity'][layer][epoch]
                    layer_sparsity_data.append({'Epoch': epoch, 'Layer': layer, 'Sparsity': sparsity})
        layer_sparsity_df = pd.DataFrame(layer_sparsity_data)
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Epoch', y='Sparsity', hue='Layer', data=layer_sparsity_df, marker='o')
        for epoch in layer_sparsity_df[layer_sparsity_df['Mask_Update']]['Epoch'].unique():
            plt.axvline(x=epoch, color='red', linestyle='--', alpha=0.5)
        plt.title(f'Sparsity Over Epochs for Generation {gen} {set_name}, {arch}')
        plt.xlabel('Epoch')
        plt.ylabel('Sparsity (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(base_dir, f'layer_sparsity_over_epochs_gen_{gen}.png'))
        plt.close()

def plot_mask_overlap(model, mask_history, base_dir, set_name, arch):
    """Plot mask overlap between different generations using Seaborn."""
    from my import percentage_overlap
    overlap_data = []
    for gen1 in mask_history:
        for gen2 in mask_history:
            if gen1 < gen2:
                prev_model = deepcopy(model)
                curr_model = deepcopy(model)
                for (name, param) in prev_model.named_parameters():
                    if name in mask_history[gen1]:
                        param.data = torch.from_numpy(mask_history[gen1][name]).to(param.device)
                for (name, param) in curr_model.named_parameters():
                    if name in mask_history[gen2]:
                        param.data = torch.from_numpy(mask_history[gen2][name]).to(param.device)
                overlap = percentage_overlap(prev_model, curr_model, percent_flag=True)
                for layer, perc in overlap.items():
                    overlap_data.append({'Layer': layer, 'Comparison': f'Gen {gen1} vs Gen {gen2}', 'Overlap': perc})
    overlap_df = pd.DataFrame(overlap_data)
    if not overlap_df.empty:
        plt.figure(figsize=(14, 6))
        sns.barplot(x='Layer', y='Overlap', hue='Comparison', data=overlap_df)
        plt.title(f'Mask Overlap Between Different Generations ({set_name}, {arch})')
        plt.xlabel('Layer')
        plt.ylabel('Overlap (%)')
        plt.xticks(rotation=45)
        plt.legend(title='Comparison')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'mask_overlap.png'))
        plt.close()
