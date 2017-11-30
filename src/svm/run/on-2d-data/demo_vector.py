import numpy

if __name__ == '__main__':
    
    x = [3, 4]

    # vector's length / magnitude / norm
    # ||x|| = sqrt(x1^2 + x2^2)
    norm = numpy.linalg.norm(x);
    print norm # 5

    # vector's direction / weights
    # w = (cos(a), cos(b)) = (x1/||x||, x2/||x||)
    # a = angle between x1 and x
    # b = angle between x2 and x
    def direction(x):
        return x / numpy.linalg.norm(x) 
    w = direction(x) # [0.6, 0.8]
    print w
    norm_w = numpy.linalg.norm(w); # always = 1 for any n-dimention vector
    print norm_w

    # dot product
    # x.y = ||x||.||y||.cos(a) = x1.y1 + x2.y2
    # a = angle between x and y
    x = [1, 2]
    y = [3, 4]
    dot = numpy.dot(x, y) # 11
    print dot

