import cv2
from typing import NamedTuple
from skimage.feature import hog
import numpy as np
from scipy.ndimage.measurements import label

class BoundingBox(NamedTuple):
    xleft: int
    xright: int
    ytop: int
    ybot: int
        
class ScaleConf(NamedTuple):
    ystart: int
    ystop: int
    scale: float

class CarFinder():
    def __init__(self, svc, X_scaler, orient, pix_per_cell, cell_per_block, img_size):
        self.svc = svc
        self.X_scaler = X_scaler
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.scale_config_list = [ScaleConf(ystart = 350,
                                            ystop = 500,
                                            scale = 1),
                                  ScaleConf(
                                            ystart = 350,
                                            ystop = 550,
                                            scale = 1.5
                                            ),
                                  ScaleConf(
                                            ystart = 350,
                                            ystop = 600,
                                            scale = 2
                                            ),
                                  ScaleConf(
                                            ystart = 450,
                                            ystop = 650,
                                            scale = 3
                                            ),
                                  ScaleConf(
                                            ystart = 500,
                                            ystop = 700,
                                            scale = 4
                                            )
                                 ]
        self.heatmap = np.zeros(shape=img_size, dtype=np.int8)
        self.cooldown_factor = 1
        self.threshold = 5
        
        
    def get_hog_features(self, img, vis=False,
                     feature_vec=False):
                         
        """
        Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
        visualization. Features will always be the first return (flattened if feature_vector= True).
        A visualization matrix will be the second return if visualize = True.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return hog(gray, orientations=self.orient, 
                   pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                   cells_per_block=(self.cell_per_block, self.cell_per_block),
                   block_norm= 'L2-Hys', transform_sqrt=True, 
                   visualize= vis, feature_vector= feature_vec)
        
    
    def find_cars(self, img, ystart, ystop, scale):
        draw_img = np.copy(img)
        bb_list = []
        img_tosearch = img[ystart:ystop,:,:]
        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, 
                                      (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            


        # Define blocks and steps as above
        nxblocks = (img_tosearch.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (img_tosearch.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pixels per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        
        # Compute individual channel HOG features for the entire image
        hog = self.get_hog_features(img_tosearch)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_features = hog[ypos:ypos+nblocks_per_window,
                                   xpos:xpos+nblocks_per_window].ravel()
     
                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell
       
                hog_features = self.X_scaler.transform(np.reshape(hog_features,(1,-1)))
                test_prediction = self.svc.predict(hog_features)
    
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                if test_prediction == 1:
                    bbox = BoundingBox(xleft=xbox_left, 
                                       ytop=ytop_draw+ystart, 
                                       ybot=ytop_draw+win_draw+ystart,  
                                       xright=xbox_left+win_draw)
                    bb_list.append(bbox)
                    cv2.rectangle(draw_img, 
                                  (xbox_left, ytop_draw+ystart), 
                                  (xbox_left+win_draw,ytop_draw+win_draw+ystart), 
                                  (0,0,255),
                                  6) 
        
        return draw_img, bb_list
        
    
    
    def find_cars_multiscale(self, img):
        bb_list = []
        for scale_conf in self.scale_config_list:
            draw_img, bb_list_part = self.find_cars(img,
                                               scale_conf.ystart,
                                               scale_conf.ystop,
                                               scale_conf.scale
                                              )
            bb_list = bb_list + bb_list_part

        return bb_list
    
    def add_heat(self, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box.ytop:box.ybot, box.xleft:box.xright] += 1
    

    def cool_down(self):
        self.heatmap[self.heatmap > 0] -= self.cooldown_factor

    def apply_threshold(self):
        # Zero out pixels below the threshold
        self.heatmap[self.heatmap <= self.threshold] = 0
        
    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
    
    def find_car_update(self, img):
        bb_list = self.find_cars_multiscale(img)
        self.add_heat(bb_list)
        self.apply_threshold()
        labels = label(self.heatmap)
        self.cool_down()
        img_bboxed = self.draw_labeled_bboxes(img, labels)
        heatmap_scaled = self.heatmap*255.0/self.threshold
        heatmap_uc = heatmap_scaled.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uc, cv2.COLORMAP_HOT)

        img_out = np.hstack((img_bboxed, heatmap_color))
        return img_out


