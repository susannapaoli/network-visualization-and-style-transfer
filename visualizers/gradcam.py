import numpy as np
import torch
from PIL import Image
from torch.autograd import Function as TorchFunc

# The ’deconvolution’ is equivalent to a backward pass through the network, except that
# when propagating through a nonlinearity, its gradient is solely computed based on the
# top gradient signal, ignoring the bottom input. In case of the ReLU nonlinearity this
# amounts to setting to zero certain entries based on the top gradient. We propose to
# combine these two methods: rather than masking out values corresponding to negative
# entries of the top gradient (’deconvnet’) or bottom data (backpropagation), we mask
# out the values for which at least one of these values is negative.


class CustomReLU(TorchFunc):
    """
    Define the custom change to the standard ReLU function necessary to perform guided backpropagation.
    We have already implemented the forward pass for you, as this is the same as a normal ReLU function.
    """

    @staticmethod
    def forward(self, x):
        output = torch.addcmul(torch.zeros(x.size()), x, (x > 0).type_as(x))
        self.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(self, y):
        ##############################################################################
        # TODO: Implement this function. Perform a backwards pass as described in    #
        # the guided backprop paper ( there is also a brief description at the top   #
        # of this page).                                                             #
        # Note: torch.addcmul might be useful, and you can access  the input/output  #
        # from the forward pass with self.saved_tensors.                             #
        ##############################################################################
        x, output = self.saved_tensors
  
        mask_1 = (x > 0).type_as(y)
        mask_2 = (y > 0).type_as(y)

        cmul_1 = torch.addcmul(torch.zeros(x.size()).type_as(x), y, mask_1)
        
        grad_input = torch.addcmul(torch.zeros(x.size()).type_as(x), cmul_1, mask_2)

        return grad_input 
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################


class GradCam:
    def guided_backprop(self, X_tensor, y_tensor, gc_model):
        """
        Compute a guided backprop visualization using gc_model for images X_tensor and
        labels y_tensor.

        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the guided backprop.

        Returns:
        - guided backprop: A numpy of shape (N, H, W, 3) giving the guided backprop for
        the input images.
        Note: Please use torch.transpose instead of torch.view or torch.permute to reshape
        the output to match the dimensions before returning.
        """

        # Thanks to Farrukh Rahman (Fall 2020) for pointing out that Squeezenet has
        #  some of it's ReLU modules as submodules of 'Fire' modules
        #
        for param in gc_model.parameters():
            param.requires_grad = True

        for idx, module in gc_model.features._modules.items():
            if module.__class__.__name__ == "ReLU":
                gc_model.features._modules[idx] = CustomReLU.apply
            elif module.__class__.__name__ == "Fire":
                for idx_c, child in gc_model.features[int(idx)].named_children():
                    if child.__class__.__name__ == "ReLU":
                        gc_model.features[int(idx)]._modules[idx_c] = CustomReLU.apply
        ##############################################################################
        # TODO: Implement guided backprop as described in paper.                     #
        # (Hint): Now that you have implemented the custom ReLU function, this       #
        # method will be similar to a single training iteration.                     #
        #                                                                            #
        # Also note that the output of this function is a numpy.                     #
        ##############################################################################
        
        output = gc_model(X_tensor)
        output_view = output.gather(1, y_tensor.view(-1, 1))
        output_view = output_view.squeeze()

        loss = output_view.sum()
        loss.backward()

        bp = X_tensor.grad.cpu().data.numpy()
        bp = np.transpose(bp, (0,2,3,1))
        return bp
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

    def grad_cam(self, X_tensor, y_tensor, gc_model):
        """
        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the gradcam.
        """
        conv_module = gc_model.features[12]
        self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.

        def gradient_hook(a, b, gradient):
            self.gradient_value = gradient[0]

        def activation_hook(a, b, activation):
            self.activation_value = activation

        conv_module.register_forward_hook(activation_hook)
        conv_module.register_backward_hook(gradient_hook)
        ##############################################################################
        # TODO: Implement GradCam as described in paper.                             #
        #                                                                            #
        # Compute a gradcam visualization using gc_model and convolution layer as    #
        # conv_module for images X_tensor and labels y_tensor.                       #
        #                                                                            #
        # Return:                                                                    #
        # If the activation map of the convolution layer we are using is (K, K) ,    #
        # student code should end with assigning a numpy of shape (N, K, K) to       #
        # a variable 'cam'. Instructor code would then take care of rescaling it     #
        # back                                                                       #
        ##############################################################################
      
        output = gc_model(X_tensor)
        output_view = output.gather(1, y_tensor.view(-1, 1))
        output_view = output_view.squeeze()

        loss = output_view.sum()
        loss.backward()


        data_grad = self.gradient_value.data
        guided_grads = (data_grad).mean((2,3), keepdim=True)
        data_act = self.activation_value.data
        cam = (data_act * guided_grads).sum(1).numpy()
  
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        # Rescale GradCam output to fit image.
        cam_scaled = []
        for i in range(cam.shape[0]):
            cam_scaled.append(
                np.array(
                    Image.fromarray(cam[i]).resize(
                        X_tensor[i, 0, :, :].shape, Image.BICUBIC
                    )
                )
            )
        cam = np.array(cam_scaled)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
