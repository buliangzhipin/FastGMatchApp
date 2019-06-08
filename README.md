# FastGMatchApp
The fastGMatch program use gpu

The original cpu program is  https://github.com/heavenstime/fastGMatch .And algorithm's introduce page is in Prof.Ymashita's homepage(http://www.ide.titech.ac.jp/~yylab/ja/). And I used cuda to accelerate it. And GPU is 10 times faster than CPU.(Nvidia 1060ti vs i7-8700K)

To compile it, you need to install cuda with visual studio and qt in visual studio and opencv.
Then you need to set all dependency just as I set it in the visual studio setting(the include path and dll) which must include cuda and opencv.
