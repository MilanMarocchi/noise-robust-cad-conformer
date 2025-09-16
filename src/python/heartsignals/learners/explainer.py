"""
    visualisation.py
    Author: Milan Marocchi

    Purpose : To visualise parts of the model, for explainability.
"""
import os
import random
import logging
import traceback
from typing import Callable, List, Optional, Tuple, Union

import PIL
import torch
from scipy.ndimage import zoom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from lime import lime_image
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d
from transformers import PreTrainedModel
from skimage.segmentation import mark_boundaries

from processing.transforms import (
    get_pil_transform, 
    get_pil_transform_numpy, 
    get_preprocess_transform, 
    get_normalise_transform
)

HERE = os.path.abspath(os.getcwd())


class ActivationsAndGradients():
    """To extract gradients and activations from models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        target_layers: List[torch.nn.Module],
        reshape_transform: Optional[Callable],
    ):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient)
            )

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            raise ValueError("Model must require grad for this to work.")

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)

            self.gradients = [grad.cpu().detach()] + self.gradients
        
        output.register_hook(_store_grad)

    def __call__(self, x, label):
        self.gradients = []
        self.activations = []
        return self.model(x, label)

    def release(self):
        for handle in self.handles:
            handle.remove()


class ClassifierOutputTarget():

    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class ClassifierOutputSoftmaxTarget():

    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]

class GradCAMPlusPlus():
    """
    Provides GradCAM++ implementation
    This focuses for 1D signals
    """

    def __init__(
            self, 
            model: PreTrainedModel, 
            target_layers: List[torch.nn.Module], 
            reshape_transform: Optional[Callable] = None,
            compute_input_gradient: bool = False,
            uses_gradients: bool = True
        ):
        self.model = model.eval()
        self.target_layers = target_layers

        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor
    ) -> torch.Tensor:
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads

        try:
            sum_activations = np.sum(activations, axis=(2,3))
        except np.exceptions.AxisError:
            sum_activations = np.sum(activations, axis=1)

        eps = 1e-6
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        try:
            weights = np.sum(weights, axis=(2,3))
        except np.exceptions.AxisError:
            weights = np.sum(weights, axis=1)

        return weights

    def get_target_width_height(
        self,
        input_tensor: torch.Tensor,
    ) -> Union[Tuple[int, int],Tuple[int,int,int],int]:
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            return depth, width, height
        elif len(input_tensor.shape) == 3:
            length = input_tensor.size(1)
            return length
        else:
            raise ValueError("Invalid tensor input size. Only supports 1D or 2D or 3D inputs currently.")

    def scale_cam_signal(
            self,
            array: Union[torch.Tensor,np.ndarray],
            shape
    ) -> np.ndarray:
        res = np.zeros(shape)
        if array.shape[0] >= shape:
            ratio = array.shape[0]/shape
            for i in range(array.shape[0]):
                res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
                if int(i/ratio) != shape-1:
                    res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
                else:
                    res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
            res = res[::-1]
            array = array[::-1]
            for i in range(array.shape[0]):
                res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
                if int(i/ratio) != shape-1:
                    res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
                else:
                    res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
            res = res[::-1]/(2*ratio)
            array = array[::-1]
        else:
            ratio = shape/array.shape[0]
            left = 0
            right = 1
            for i in range(shape):
                if left < int(i/ratio):
                    left += 1
                    right += 1
                if right > array.shape[0]-1:
                    res[i] += array[left]
                else:
                    res[i] += array[right] * \
                        (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
            res = res[::-1]
            array = array[::-1]
            left = 0
            right = 1
            for i in range(shape):
                if left < int(i/ratio):
                    left += 1
                    right += 1
                if right > array.shape[0]-1:
                    res[i] += array[left]
                else:
                    res[i] += array[right] * \
                        (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
            res = res[::-1]/2
            array = array[::-1]
        return res

    def get_cam_signal(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> torch.Tensor:
        activations = activations[0, :]
        grads = grads[0, :]
        weights = self.get_cam_weights(input_tensor, target_layer, target_layers, activations, grads)
        cam = activations.T.dot(weights)

        return cam


    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        eigen_smooth: bool
    ):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []

        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_gradients = None

            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_gradients = grads_list[i]
            
            cam = self.get_cam_signal(input_tensor, target_layer, target_layers, layer_activations, layer_gradients, eigen_smooth) # type: ignore
            scaled = self.scale_cam_signal(cam, target_size)
            cam_per_target_layer.append(scaled)

        return cam_per_target_layer


    def aggregate_multi_layers(
        self,
        cam_per_target_layer: list[np.ndarray],
        shape
    ) -> np.ndarray:
    
        cam_per_target_layer = np.stack(cam_per_target_layer, axis=1)
        if cam_per_target_layer.ndim > 1:
            result = np.mean(cam_per_target_layer, axis=1)
        else:
            result = cam_per_target_layer

        return self.scale_cam_signal(result, shape)

    def forward(
        self,
        input_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool = False,
    ):
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor, label_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories] # type: ignore

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True) # type: ignore

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer, input_tensor.size(1))

    def __call__(
        self,
        input_tensor: torch.Tensor,
        label_tensor: torch.Tensor,
        targets: Optional[List[torch.nn.Module]] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        return self.forward(input_tensor, label_tensor, targets, eigen_smooth) # type: ignore
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            stack_trace = traceback.format_exc()
            print(f"An exception occured in the cam block: {exc_type}, Message: {exc_value}")
            print(f"Stack trace: {stack_trace}")
            return True

    def __del__(self):
        self.activations_and_grads.release()
    
    def __enter__(self):
        return self


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    """
    Visualise some outputs of a specific model
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


"""
Hooks: To be added to models for explainability
"""
# Globals for the hooks
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    """
    Backwards hook to get the gradients for explainability
    """
    global gradients
    logging.debug('Backward hook running...')
    gradients = grad_output
    logging.debug(f'Gradients size: {gradients[0].size()}')


def forward_hook(module, args, output):
    """
    Forwards hook to get the activations for explainability
    """
    global activations
    logging.debug('Forward hook running...')
    activations = output
    logging.debug(f'Activations size: {activations.size()}')


def relu_hook_function(module, grad_in, grad_out):
    """
    Hook for guided backprop
    """
    if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.0),)
    elif isinstance(module, BasicConv2d):
        return (torch.clamp(grad_in[0], min=0.0),)
"""
Hooks
"""

class Explainer():
    """
    Class to explain a model
    """

    def __init__(self, data_dir, model, model_code):
        self.data_dir = data_dir
        self.model_ft = model.model_ft
        self.model_code = model_code

    def get_explain_images(self):
        return None, None

    def explain(self):
        abnormal_images, normal_images = self.get_explain_images()

        if abnormal_images is not None and normal_images is not None:
            # Normal images
            logging.info(f"normal_images: {normal_images}")
            for path in normal_images:
                img = None
                img = PIL.Image.open(path).convert('RGB')

                self.lime_explain(img, 0)
                self.grad_cam_explain(img)
                self.saliency_explain(img)

            # Abnormal images
            logging.info(f"abnormal_images: {abnormal_images}")
            for path in abnormal_images:
                img = None
                img = PIL.Image.open(path).convert('RGB')

                self.lime_explain(img, 1)
                self.grad_cam_explain(img)
                self.saliency_explain(img)

    def batch_predict(self):
        """
        Predicts outputs of a batch for a specific model
        """
        def inner_batch_predict(images):
            """
            This inner function is to be used for utilities like lime that expect just one input
            NOTE: It is expected that images will be a numpy array
            """
            pre_process_transform = get_normalise_transform(size=299)

            # Pre-process data to be correct format
            images = np.transpose(images, (0, 3, 1, 2))
            images = torch.from_numpy(images).float()

            self.model_ft.eval()
            batch = torch.stack(tuple(pre_process_transform(i) for i in images), dim=0)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model_ft.to(device)
            batch = batch.to(device)

            outputs = self.model_ft(batch)
            probs = F.softmax(outputs, dim=1)

            return probs.detach().cpu().numpy()

        return inner_batch_predict


    def visualise_lime_explaination(self, image, mask):
        """
        Visualises a lime explaination
        """
        img_boundry = mark_boundaries(image, mask)
        plt.imshow(img_boundry)
        plt.tight_layout()
        plt.show()


    def lime_explain(self, img, label):
        """
        Uses lime to explain a normal and abnormal image classification
        """
        pil_transform = get_pil_transform_numpy(size=299)

        batch_predict_func = self.batch_predict()

        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(np.transpose(pil_transform(img), (1, 2, 0)),
                                                batch_predict_func,
                                                top_labels=2)
        print(label, explaination.top_labels)

        temp, mask = explaination.get_image_and_mask(label,
                                                    positive_only=True,
                                                    num_features=5,
                                                    hide_rest=False
                                                    )
        self.visualise_lime_explaination(temp, mask)

        temp, mask = explaination.get_image_and_mask(label,
                                                    positive_only=False,
                                                    num_features=10,
                                                    hide_rest=False
                                                    )
        self.visualise_lime_explaination(temp, mask)


    def grad_cam_explain(self, image):
        """
        Uses the gradCAM approach to exaplain a normal and abormal image classification
        """

        # Preprocess input
        preprocess_transform = get_preprocess_transform(size=299)
        img = preprocess_transform(image).requires_grad_()

        # Convert model to cpu for this part
        self.model_ft.to(torch.device("cpu"))
        self.model_ft.eval()

        # Apply hooks
        if self.model_code == "resnet":
            layers = [self.model_ft.layer4]
        elif self.model_code == "vgg":
            layers = [self.model_ft.features]
        elif self.model_code == "inception":
            layers = [self.model_ft.Mixed_7c]

        #for i, module in enumerate(self.model_ft.modules()):
        #    if isinstance(module, BasicConv2d):
        #        layer = module


        #layer.register_full_backward_hook(backward_hook, prepend=False)
        #layer.register_forward_hook(forward_hook, prepend=False)
        backwards_hooks = [layer.register_full_backward_hook(backward_hook, prepend=False)
                          for layer in layers]
        forwards_hooks = [layer.register_forward_hook(forward_hook, prepend=False)
                         for layer in layers]

        # Run to get gradients and activations
        out = self.model_ft(img.unsqueeze(0))
        out_max_index = torch.argmax(out)
        out_max = out[0, out_max_index]
        out_max.backward()

        # Run GradCAM algorithm
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # Show figure
        plt.figure()
        pil_transform = get_pil_transform(size=299)
        image = pil_transform(image)
        plt.imshow(to_pil_image(image, mode="RGB"))

        overlay = to_pil_image(heatmap.detach(), mode='F').resize(
            ((299, 299)), resample=PIL.Image.BICUBIC
        )
        cmap = colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        plt.imshow(overlay, alpha=0.4, interpolation='nearest')
        plt.show()

        # Clean up
        [backwards_hook.remove() for backwards_hook in backwards_hooks]
        [forwards_hook.remove() for forwards_hook in forwards_hooks]


    def saliency_explain(self, image):
        """
        Uses the saliency map approach to explain a normal and abnormal image classification
        """
        # Preprocess input
        preprocess_transform = get_preprocess_transform(size=299)
        img = preprocess_transform(image).requires_grad_()

        # Setup model
        self.model_ft.eval()
        self.model_ft.to(torch.device("cpu"))

        # Setting to be guided so that it prevents backward flow of negative gradients on ReLU
        # NOTE: This is done through a hook so need to set this up
        for i, module in enumerate(self.model_ft.modules()):
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_hook_function)
            elif isinstance(module, BasicConv2d):
                module.register_backward_hook(relu_hook_function)

        # Run to get gradients and activations
        out = self.model_ft(img.unsqueeze(0))
        out_max_index = torch.argmax(out)
        out_max = out[0, out_max_index]
        out_max.backward()

        saliency = img.grad.data

        # Display saliency map as a heatmap
        plt.figure()
        plt.imshow(np.abs(saliency.numpy()).max(axis=0), cmap=colormaps['jet'])
        plt.show()


class ModelExplainer(Explainer):

    def __init__(self, data_dir, model, model_code):
        super().__init__(data_dir, model, model_code)

    def get_explain_images(self):
        abnormal_dir = os.path.join(HERE, "image_datasets", self.data_dir, "test", "abnormal")
        normal_dir = os.path.join(HERE, "image_datasets", self.data_dir, "test", "normal")

        normal_images = [os.path.join(normal_dir, x)
                         for x in ["a0189:-1:0.png", "a0155:-1:0.png"]]
        abnormal_images = [os.path.join(abnormal_dir, x)
                           for x in ["a0005:1:0.png", "a0057:1:0.png"]]


        return abnormal_images, normal_images

class EnsembleModelExplainer(Explainer):

    def __init__(self, data_dir, model, model_code, ensemble):
        self.data_dir = data_dir
        self.models = model.model_ft.models
        self.model = model.model_ft
        self.model_code = model_code
        self.ensemble = ensemble

    def batch_predict(self, idx):
        """
        Predicts outputs of a batch for a specific model
        """
        def inner_batch_predict(images):
            """
            This inner function is to be used for utilities like lime that expect just one input
            NOTE: It is expected that images will be a numpy array
            """
            pre_process_transform = get_normalise_transform(size=299)

            # Pre-process data to be correct format
            images = np.transpose(images, (0, 3, 1, 2))
            images = torch.from_numpy(images).float()

            model_ft = self.models[idx]

            model_ft.eval()
            batch = torch.stack(tuple(pre_process_transform(i) for i in images), dim=0)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_ft.to(device)
            batch = batch.to(device)

            outputs = model_ft(batch)
            probs = F.softmax(outputs, dim=1)

            return probs.detach().cpu().numpy()

        return inner_batch_predict

    def lime_explain(self, img, label, idx):
        """
        Uses lime to explain a normal and abnormal image classification
        """
        pil_transform = get_pil_transform_numpy(size=299)

        batch_predict_func = self.batch_predict(idx)

        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(np.transpose(pil_transform(img), (1, 2, 0)),
                                                batch_predict_func,
                                                top_labels=2)
        print(label, explaination.top_labels)

        temp, mask = explaination.get_image_and_mask(label,
                                                    positive_only=True,
                                                    num_features=5,
                                                    hide_rest=False
                                                    )
        self.visualise_lime_explaination(temp, mask)

        temp, mask = explaination.get_image_and_mask(label,
                                                    positive_only=False,
                                                    num_features=10,
                                                    hide_rest=False
                                                    )
        self.visualise_lime_explaination(temp, mask)


    def grad_cam_explain(self, image, idx):
        """
        Uses the gradCAM approach to exaplain a normal and abormal image classification
        """

        # Preprocess input
        preprocess_transform = get_preprocess_transform(size=299)
        img = preprocess_transform(image).requires_grad_()

        # Convert model to cpu for this part
        model_ft = self.models[idx]
        model_ft.to(torch.device("cpu"))
        model_ft.eval()

        # Apply hooks
        if self.model_code == "resnet":
            layers = [model_ft.layer4]
        elif self.model_code == "vgg":
            layers = [model_ft.features]
        elif self.model_code == "inception":
            layers = [model_ft.Mixed_7c]

        #for i, module in enumerate(self.model_ft.modules()):
        #    if isinstance(module, BasicConv2d):
        #        layer = module
        #        print(i)


        #layer.register_full_backward_hook(backward_hook, prepend=False)
        #layer.register_forward_hook(forward_hook, prepend=False)
        backwards_hooks = [layer.register_full_backward_hook(backward_hook, prepend=False)
                          for layer in layers]
        forwards_hooks = [layer.register_forward_hook(forward_hook, prepend=False)
                         for layer in layers]

        # Run to get gradients and activations
        out = model_ft(img.unsqueeze(0))
        out_max_index = torch.argmax(out)
        out_max = out[0, out_max_index]
        out_max.backward()

        # Run GradCAM algorithm
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # Show figure
        plt.figure()
        pil_transform = get_pil_transform(size=299)
        image = pil_transform(image)
        plt.imshow(to_pil_image(image, mode="RGB"))

        overlay = to_pil_image(heatmap.detach(), mode='F').resize(
            ((299, 299)), resample=PIL.Image.BICUBIC
        )
        cmap = colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        plt.imshow(overlay, alpha=0.4, interpolation='nearest')
        plt.show()

        # Clean up
        [backwards_hook.remove() for backwards_hook in backwards_hooks]
        [forwards_hook.remove() for forwards_hook in forwards_hooks]


    def saliency_explain(self, image, idx):
        """
        Uses the saliency map approach to explain a normal and abnormal image classification
        """
        # Preprocess input
        preprocess_transform = get_preprocess_transform(size=299)
        img = preprocess_transform(image).requires_grad_()

        # Setup model
        model_ft = self.models[idx]
        model_ft.eval()
        model_ft.to(torch.device("cpu"))

        # Setting to be guided so that it prevents backward flow of negative gradients on ReLU
        # NOTE: This is done through a hook so need to set this up
        for i, module in enumerate(model_ft.modules()):
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_hook_function)
            elif isinstance(module, BasicConv2d):
                module.register_backward_hook(relu_hook_function)

        # Run to get gradients and activations
        out = model_ft(img.unsqueeze(0))
        out_max_index = torch.argmax(out)
        out_max = out[0, out_max_index]
        out_max.backward()

        saliency = img.grad.data

        # Display saliency map as a heatmap
        plt.figure()
        plt.imshow(np.abs(saliency.numpy()).max(axis=0), cmap=colormaps['jet'])
        plt.show()

    def get_weightings(self):
        """
        Finds the weightings between models in the ensemble
        """
        return self.model.classifier.weight

    def get_explain_images(self):
        abnormal_paths = [os.path.join(HERE, "image_datasets", self.data_dir, str(x), "test", "abnormal")
                          for x in range(int(self.ensemble))]
        normal_paths = [os.path.join(HERE, "image_datasets", self.data_dir, str(x), "test", "normal")
                        for x in range(int(self.ensemble))]

        abnormal_images = [[os.path.join(path, x)
                            for x in ["a0005:1:0.png", "a0057:1:0.png"]] for path in abnormal_paths]
        normal_images = [[os.path.join(path, x)
                          for x in ["a0189:-1:0.png", "a0155:-1:0.png"]] for path in normal_paths]

        return abnormal_images, normal_images

    def explain(self):
        abnormal_images, normal_images = self.get_explain_images()

        if abnormal_images is not None and normal_images is not None:
            # Normal images
            logging.info(f"normal_images: {normal_images}")
            for paths in normal_images:
                for idx, path in enumerate(paths):
                    img = None
                    img = PIL.Image.open(path).convert('RGB')

                    self.lime_explain(img, 0, idx)
                    self.grad_cam_explain(img, idx)
                    self.saliency_explain(img, idx)

            # Abnormal images
            logging.info(f"abnormal_images: {abnormal_images}")
            for paths in abnormal_images:
                for idx, path in enumerate(paths):
                    img = None
                    img = PIL.Image.open(path).convert('RGB')

                    self.lime_explain(img, 1, idx)
                    self.grad_cam_explain(img, idx)
                    self.saliency_explain(img, idx)

        # Find the weights for the final layer
        print(f"Weights of combination between models: {self.get_weightings()}")

class ExplainerFactory():

    def __init__(self, data_dir, model, model_code, ensemble):
        self.model = model
        self.data_dir = data_dir
        self.model_code = model_code
        self.ensemble = ensemble

    def create(self, explain):
        if not explain:
            return Explainer(self.data_dir, self.model, self.model_code)

        if self.ensemble is not None:
            return EnsembleModelExplainer(self.data_dir, self.model, self.model_code, self.ensemble)
        else:
            return ModelExplainer(self.data_dir, self.model, self.model_code)
