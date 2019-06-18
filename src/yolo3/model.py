"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))   #compose函数相当于实现复合函数f(g(x))

#获得一个resblock单元，包含1+num_blocks*2个卷积层
def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):   #通过该函数获得了darknet主体的52层卷积层
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):  #卷积核尺寸均为（1，1）和（3，3）交替
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""  #将得到的某layer的特征图转换为bbox坐标，长宽，置信度和类别概率值
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
	#将anchors形状由[N,2]变为anchors_tensor-->[1,1,1,N,2]

    grid_shape = K.shape(feats)[1:3] # height, width  #feats即为yolo_output[l]。grid_shape即为特征图的宽、高
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])    #K.arange(0, stop=grid_shape[0])再K.reshape产生张量[[[[0]]],[[[1]]],[[[2]]],....[[[特征图的高-1]]]]，再经过tile变为[[[[0]],[[0]],[[0]],...[[特征图的宽-1]]],[[[1]],[[1]],[[1]],...[[特征图的宽-1个]]],[[[2]]，[[2]]，[[2]],...[[宽-1个]]],....[[[特征图的高-1]],[[特征图的高-1]],[[特征图的高-1]],...[[特征图的高-1]]]](第一阶为特征图的高-1)
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])    #K.tile(K.reshape(K.arange(0,stop=grid_shape[1]),[1,-1,1,1]),[grid_shape[0], 1, 1, 1])得到[[[[0]],[[1]],[[2]],[[特征图的宽-1]]],[[[0]],[[1]],[[2]],[[特征图的宽-1]]],[[[0]],[[1]],[[2]],[[特征图的宽-1]]],...[[[0]],[[1]],[[2]],[[特征图的宽-1]]]](第一阶为特征图的高-1)
    grid = K.concatenate([grid_x, grid_y])  #将以上得到的两个张量合并
    grid = K.cast(grid, K.dtype(feats))    #

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])   #feats变形为[？？？,特征图的高,特征图的宽,anchor数,5+类别数]

    # Adjust preditions to each spatial grid point and anchor size. ###此即为论文中的一组公式，得到bx,by,bw,bh
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])  #feats第5维即为置信度
    box_class_probs = K.sigmoid(feats[..., 5:])  #第五维之后即为类别概率值

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]    #box_xy为bbox框中心坐标
    box_hw = box_wh[..., ::-1]    #
    input_shape = K.cast(input_shape, K.dtype(box_yx))  #input_shape为[416,416]
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape    #计算出offset值
    scale = input_shape/new_shape    #计算出scale值
    box_yx = (box_yx - offset) * scale   #计算出乘以scale后的box
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)   #bbox坐上角点坐标
    box_maxes = box_yx + (box_hw / 2.)  #bbox右下角点坐标
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])   #将bbox还原为原图比例
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])   #boxes形状变为[N,4]
    box_scores = box_confidence * box_class_probs    #计算每一类别的预测得分
    box_scores = K.reshape(box_scores, [-1, num_classes])  #box_scores形状为[N,num_classes]
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)    #输入yolo_outputs为三层layer的输入
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)   #存储各层对应的boxes[N,4]和box_scores[N,num_classes]，此时boxes和box_scores均为list，各元素即为各layer得到的boxes和box_scores
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)  #将存储各层对应的boxes和和box_scores中各元素合并
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold  #mask形状为[N,num_classes],若某class score>thresh则对应mask[:,c]=1，反之=0
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])  # tf.boolean_mask(a,b) 将使a (m维)矩阵仅保留与b中“True”元素同下标的部分，并将结果展开到m-1维
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])  #这两句通过boolean_mask()函数滤除box和box_scores中未检测该类别c的bbox所对应的行
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)   #删除掉那些与之前的选择的边框具有很高的IOU的边框。边框是以[y1,x1,y2,x2],(y1,x1)和(y2,x2)是边框的对角坐标，当然也可以提供被归一化的坐标。返回的是被选中的那些留下来的边框在参数boxes里面的下标位置。
        class_boxes = K.gather(class_boxes, nms_index)           #那么你可以使用tf.gather的操作或者利用keras.backend的gather函数来从参数boxes来获取选中的边框。
        class_box_scores = K.gather(class_box_scores, nms_index)   #NMS用于移除同一物体检测出的多余的边界框
        classes = K.ones_like(class_box_scores, 'int32') * c      #此时class_boxes为选出的最优检测bbox(具有最大pc值)，class_box_scores为边界框对应的分数
        boxes_.append(class_boxes)            #K.ones_like()产生和输入tensor相同形状和类型，元素全为1的tensor
        scores_.append(class_box_scores)
        classes_.append(classes)  #合并各类别对应的classes[M,1],得到[](元素值为对应类别C在num_classes中的下标)
    boxes_ = K.concatenate(boxes_, axis=0)   #将各类别检测出的最优bbox[M,4]进行合并得到[num_classes,M,4]
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)  #m为图片张数，T为一张图片的box数，5个元素分别为：x1,y1,x2,y2,class_id
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32  ###input_shape是（416X416）
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]  #anchor_mask[l]为第l个layer对应的anchor_mask

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2   #计算true_boxes张量中每个box的中心坐标
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  #计算true_boxes每个box对应的宽高
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]  #将true_boxes张量每个box的前两个坐标改为box_xy/416
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]  #将每个box的第3,4个坐标改为该box_wh/416

    m = true_boxes.shape[0]   #m为图片数
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]  #对于三个layer，网格形状分别为：（416/32）^2即（13X13）,(416/16)^2和（52X52）
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]     #产生y_true张量，形状为[m,网格宽，网格高，anchor个数，5(代表Pc,bx,by,bw,bh)+(c1,c2,c3,...)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  #将anchors形状由[N,2]变为[1,N,2]
    anchor_maxes = anchors / 2.   #
    anchor_mins = -anchor_maxes   #没看懂？？
    valid_mask = boxes_wh[..., 0]>0  #滤除所有宽<0的box

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]   #得到第b张图片对应的所有box的wh，即wh张量为[T,2]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)  #改变wh形状为[1,T,2]
        box_maxes = wh / 2.  #同上，/2什么意思，没看懂？？
        box_mins = -box_maxes
		
        intersect_mins = np.maximum(box_mins, anchor_mins)   
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)  #以上为对该张图片通过IOU寻找最佳anchor

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)   #对该张图片的每个bounding_boxes找到形状最相似的anchor（即IOU最大的anchor）

        for t, n in enumerate(best_anchor):   #t为该张图片某一bounding_box对应最大IOU的anchors的序号，n为对应最大的
            for l in range(num_layers):
                if n in anchor_mask[l]:  #n是否在anchor_mask[l]中
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')   #np.floor返回不大于输入参数的最大整数
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')   #此时的true_boxes[b,t,0]代表b张图片,与anchor有着最大IOU的box的序号t的box_xy/416（xy对应01）
                    k = anchor_mask[l].index(n)   #接上，再乘以某层的grid_shape即为该bounding_box的中心坐标/416再*grid_shape, 即I,J为该bounding_box中心坐标究竟落入了网格图中的哪一个grid的序号
                    c = true_boxes[b,t, 4].astype('int32')   #k为n对应的anchor_mask的序号，c为这个bounding_box对应的类别标签
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4] #将该true_boxes中的bounding_boxes的对应元素赋给y_true对应元素
                    y_true[l][b, j, i, k, 4] = 1  #将y_true中的该bbox设置为包含object
                    y_true[l][b, j, i, k, 5+c] = 1  #将该bbox的类别（c对应的类别序号）置为1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]  #输入args为一个list，前num_layers各元素为yolo_outputs，之后
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]    #即为Pc值，0/1
        true_class_probs = y_true[l][..., 5:]   #即为所有的类别概率值

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
