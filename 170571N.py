import cv2
import numpy as np
import math

def get_zero_array(W,H):
    zero_array = [([0]*H) for p in range(W)]
    return zero_array

def convolution(img, mask):
    size = len(mask)
    W = img.shape[0]-(size-1)
    H = img.shape[1]-(size-1)
    img_new = get_zero_array(W,H)
    for i in range(W):
        for j in range(H):
            x = 0
            for k in range(size):
                for l in range(size):
                    x = x + img[i+k][j+l]*mask[k][l]
            img_new[i][j]=x
    return img_new

def gaussian_filter(kernel_size,sigma):
    gau_filter = np.zeros((kernel_size , kernel_size))
    m = kernel_size//2
    n = kernel_size//2
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gau_filter[x+m, y+n] = (1/x1)*x2
    return gau_filter


def sobel_edge_detection(img_new):
    sobl_filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    img_new_x  = convolution(np.array(img_new), sobl_filter_x)
    sobl_filter_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
    img_new_y  = convolution(np.array(img_new), sobl_filter_y)
    return img_new_x,img_new_y

def gradient_strength(img_x,img_y):
    gradient_strength = np.sqrt(np.square(np.array(img_x)) + np.square(np.array(img_y)))
    gradient_strength *= 255.0 / gradient_strength.max()
    return gradient_strength


def gradient_direction(img_x,img_y):
    gradient_direction_list = []
    for x in range(len(img_x)):
        grad_list = []
        for x_ in range(len(img_x[0])):
            val = math.atan2(img_x[x][x_], img_y[x][x_])
            val = math.degrees(val)
            val+=180
            grad_list.append(val)
        gradient_direction_list.append(grad_list)   
    return np.array(gradient_direction_list)

def non_maxima_suppression(gradient_strength_list,gradient_direction_list):
    row, col = gradient_strength_list.shape
    suppression_img = get_zero_array(row,col)
    pi = 180

    for x in range(1, row - 1):
        for y in range(1, col - 1):
            gr_dir = gradient_direction_list[x, y]

            if (0 <= gr_dir < pi / 8) or (15 * pi / 8 <= gr_dir <= 2 * pi):
                pixel_x = gradient_strength_list[x, y - 1]                
                pixel_y = gradient_strength_list[x, y + 1]

            elif (pi / 8 <= gr_dir < 3 * pi / 8) or (9 * pi / 8 <= gr_dir < 11 * pi / 8):
                pixel_x = gradient_strength_list[x + 1, y - 1]
                pixel_y = gradient_strength_list[x - 1, y + 1]

            elif (3 * pi / 8 <= gr_dir < 5 * pi / 8) or (11 * pi / 8 <= gr_dir < 13 * pi / 8):
                pixel_x = gradient_strength_list[x - 1, y]
                pixel_y = gradient_strength_list[x + 1, y]

            else:
                pixel_x = gradient_strength_list[x - 1, y - 1]
                pixel_y = gradient_strength_list[x + 1, y + 1]

            if gradient_strength_list[x, y] >= pixel_x and gradient_strength_list[x, y] >= pixel_y:
                suppression_img[x][y] = gradient_strength_list[x, y]
    return suppression_img

def final_image(suppression_img,weak, low , high):
    strong = 255
    row_, col_ = len(suppression_img),len(suppression_img[0])
    threshold_img = get_zero_array(row_,col_)
    threshold_img = np.array(threshold_img)
    strong_row, strong_col = np.where(np.array(suppression_img) >= high)
    weak_row, weak_col = np.where((np.array(suppression_img) <= high) & (np.array(suppression_img) >= low))
    threshold_img[strong_row, strong_col] = strong
    threshold_img[weak_row, weak_col] = weak
    return threshold_img

if __name__ == '__main__':
    print("........Read Image..........")
    image_name = input("Enter your image name for edge detection (should be save in the same location of the python file) like image_name.png/.jpg: ")
    img = cv2.imread(image_name,0) # open image in gray scale

    # Gaussian filter for noise filtering
    print("........Start Gaussain Filtering..........")
    kernel_size = int(input("Enter your Gaussian Kernel Size: "))
    sigma = int(input("Enter Sigma value for Gaussian Filer : "))
    kernel = gaussian_filter(kernel_size,sigma)
    gaussian_img = convolution(img ,kernel)

    # Sobel edge detection for blured Image
    print("........Sobel edge detection for blured Image..........")
    img_new_x,img_new_y = sobel_edge_detection(gaussian_img)

    # find Gradient Direction and Gradient strength
    print("........find Gradient Direction and Gradient strength..........")
    gradient_direction_list = gradient_direction(img_new_x,img_new_y)
    gradient_strength_list = gradient_strength(img_new_x,img_new_y)

    # find Non maxima Suppression
    print("........Non maxima Suppression..........")
    non_maxima_suppression_list = non_maxima_suppression(gradient_strength_list,gradient_direction_list)
    

    # Double Threshold 
    print("........Double Threshold..........")
    low = int(input("Enter your low Threshold Value : "))
    high = int(input("Enter your high Threshold Value : "))
    weak= 50
    output = final_image(non_maxima_suppression_list,weak, low , high)
    if cv2.imwrite('edge_detected_image.jpg',output) == True:
        cv2.imwrite('edge_detected_image.jpg',output)
        print("final image is save in your code location!")
    else:
        print("save failed!!!!!!!!!")