import pygame
import torch
import numpy as np
import sys

class NeuralNetworkVisualizer:
    def __init__(self, model, surface,\
                        nodes_radius=40, nodes_color='white', \
                        border=True, nodes_border_color='black',\
                        weight_multiple=7, border_width=2, 
                        min_weight_thickness=2,
                        top_padding=0, bottom_padding=0,\
                        left_padding=0, right_padding=0):
        pygame.init()
        self.screen = surface
        self.surfrect = self.screen.get_rect()
        self.width = self.surfrect.w
        self.heigth = self.surfrect.h
        self.clock = pygame.time.Clock()
        
        self.model = model
        self.max_neurons = self.get_max_neurons_number()
        
        nodes_diameter = self.heigth // self.max_neurons
        nodes_radius_ = nodes_diameter // 2
        self.node_radius = min(nodes_radius, nodes_radius_)
        self.weight_multiple = weight_multiple
        self.min_weight_thickness = min_weight_thickness
        self.nodes_color = nodes_color
        self.border = border
        self.border_width = border_width
        self.nodes_border_color = nodes_border_color
        self.top_padding = max(self.node_radius, top_padding)
        self.bottom_padding = max(self.node_radius, bottom_padding)
        self.right_padding = max(self.node_radius, right_padding)
        self.left_padding = max(self.node_radius, left_padding)
        
        self.layers_sizes = self.get_layer_sizes()
        self.layers_nodes_y_pos = self.get_layers_nodes_y_pos()
        self.layers_x_pos = self.get_layers_x_pos()
        
        if False:
            print(self.layers_nodes_y_pos)
            print()
            print(self.layers_x_pos)
            print()
            print(self.layers_sizes)
            print()
            print(self.max_neurons)
            print()
        
    def get_max_neurons_number(self):
        max_neurons = 0
        
        for layer in self.model.children():
            if isinstance(layer, torch.nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                if in_features > max_neurons:
                    max_neurons = in_features
                if out_features > max_neurons:
                    max_neurons = out_features
                    
        return max_neurons
    
    def get_layers_nodes_y_pos(self):
        layer_nodes_y = {}
        layer_index = 0
        
        # Collect layer information
        for layer in self.model.children():
            if isinstance(layer, torch.nn.Linear):
                if layer_index == 0:
                    layer_nodes_y[layer_index] = np.linspace(self.top_padding, self.heigth - self.bottom_padding, self.max_neurons)
                    layer_index += 1
                layer_nodes_y[layer_index] = np.linspace(self.top_padding, self.heigth - self.bottom_padding, self.max_neurons)
                layer_index += 1
                
        return layer_nodes_y
    
    def get_layers_x_pos(self):
        x_positions = np.linspace(self.left_padding, self.width - self.right_padding, len(self.layers_sizes))
        return x_positions
    
    def get_layer_sizes(self):
        layer_sizes = []
        
        # Collect layer information
        for layer in self.model.children():
            if isinstance(layer, torch.nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                
                if len(layer_sizes) == 0:
                    layer_sizes.append(in_features)
                layer_sizes.append(out_features)
        return layer_sizes
    
    def get_layers_weight(self):
        layers_weight = []
        layer_id = 0
        for layer in  self.model.children():
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().numpy()
                layers_weight.append(weights)
                
        return layers_weight
    
    def save_as_image(self, filename):
        self.screen.fill('white')
        self.draw()
        pygame.image.save(self.screen, filename)
    
    def draw_nodes(self):
        for layer_id, nodes_y in self.layers_nodes_y_pos.items():
            x = self.layers_x_pos[layer_id]
            layer_size = self.layers_sizes[layer_id]
            additional_neurons = self.max_neurons - layer_size
            start_idx = additional_neurons // 2
            skip_nuerons = additional_neurons - start_idx
            stop_idx = self.max_neurons - skip_nuerons
            
            for i, y in enumerate(nodes_y):
                if i < start_idx:
                    continue
                if i >= stop_idx:
                    break
                
                if self.border:
                    pygame.draw.circle(self.screen, self.nodes_border_color, (int(x), int(y)), self.node_radius)
                    pygame.draw.circle(self.screen, self.nodes_color, (int(x), int(y)), self.node_radius - self.border_width)
                else:
                    pygame.draw.circle(self.screen, self.nodes_color, (int(x), int(y)), self.node_radius)

    def draw_connections(self):
        if True:
            if True:
                layers_weights = self.get_layers_weight()
                layers_nodes_y_pos = self.layers_nodes_y_pos
                
                for layer_idx, layer_weight in enumerate(layers_weights):
                    # layer_weight shape is (layer_out_features, layer_in_features)
                    prev_layer_size = len(layer_weight[0])
                    curr_layer_size = len(layer_weight)
                    prev_layer_x_pos = self.layers_x_pos[layer_idx]
                    curr_layer_x_pos = self.layers_x_pos[layer_idx + 1]
                    prev_layer_nodes_y_pos = layers_nodes_y_pos.get(layer_idx)
                    curr_layer_nodes_y_pos = layers_nodes_y_pos.get(layer_idx + 1)
                    
                    if False:
                        print('layer_idx, prev_layer_size, curr_layer_size, prev_layer_x_pos, curr_layer_x_pos, prev_layer_nodes_y_pos, curr_layer_nodes_y_pos')
                        print(layer_idx, prev_layer_size, curr_layer_size, prev_layer_x_pos, curr_layer_x_pos, prev_layer_nodes_y_pos, curr_layer_nodes_y_pos)
                        print()
                    
                    for prev_node in range(prev_layer_size):
                        for curr_node in range(curr_layer_size):
                            weight = layer_weight[curr_node, prev_node]
                            if weight != 0:
                                prev_layer_additional_neurons = self.max_neurons - prev_layer_size
                                curr_layer_additional_neurons = self.max_neurons - curr_layer_size
                                prev_layer_start_idx = prev_layer_additional_neurons // 2
                                curr_layer_start_idx = curr_layer_additional_neurons // 2
                                
                                prev_node_pos = int(prev_layer_x_pos), int(prev_layer_nodes_y_pos[prev_layer_start_idx + prev_node])
                                curr_node_pos = int(curr_layer_x_pos), int(curr_layer_nodes_y_pos[curr_layer_start_idx + curr_node])
                                
                                thickness = abs(weight) * self.weight_multiple
                                
                                if thickness < self.min_weight_thickness:
                                    brightness = lerp(10, 255, thickness / self.min_weight_thickness)
                                    thickness = self.min_weight_thickness
                                else:
                                    brightness = 255
                                    thickness = int(thickness)
                                #print(abs(weight), abs(weight) * self.weight_multiple, thickness, brightness)
                                color = (0, 0, 255, brightness) if weight > 0 else (255, 0, 0, brightness)
                                
                                pygame.draw.line(self.screen, color, prev_node_pos, curr_node_pos, thickness)
                
    
    def draw(self):
        self.draw_connections()
        self.draw_nodes()

def lerp(a, b, weight):
    if weight >= 1:
        return b
    return a + (b-a) * weight

# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    import torch.nn.init as init
    
    # Define a simple model for demonstration
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 7),
        torch.nn.ReLU(),
        torch.nn.Linear(7, 10),
        torch.nn.Linear(10, 6),
        torch.nn.Linear(6, 1),
        torch.nn.Linear(1, 4)
    )
    surface = pygame.display.set_mode((720, 1472))
    clock = pygame.time.Clock()
    visualizer = NeuralNetworkVisualizer(model, surface)
    
    # Helper function to reinitialize weights
    def reinitialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


    
    clicked = False
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                clicked = True
                reinitialize_weights(visualizer.model)
            elif e.type == pygame.MOUSEBUTTONUP:
                clicked = False
                
        surface.fill('white')
        visualizer.draw()
        pygame.display.flip()
        clock.tick(60)
        