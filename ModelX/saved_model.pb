╢Т
ш╛
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-0-ga4dfb8d1a718Ьн
j
Adamax/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdamax/iter
c
Adamax/iter/Read/ReadVariableOpReadVariableOpAdamax/iter*
_output_shapes
: *
dtype0	
n
Adamax/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_1
g
!Adamax/beta_1/Read/ReadVariableOpReadVariableOpAdamax/beta_1*
_output_shapes
: *
dtype0
n
Adamax/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/beta_2
g
!Adamax/beta_2/Read/ReadVariableOpReadVariableOpAdamax/beta_2*
_output_shapes
: *
dtype0
l
Adamax/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamax/decay
e
 Adamax/decay/Read/ReadVariableOpReadVariableOpAdamax/decay*
_output_shapes
: *
dtype0
|
Adamax/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdamax/learning_rate
u
(Adamax/learning_rate/Read/ReadVariableOpReadVariableOpAdamax/learning_rate*
_output_shapes
: *
dtype0
Ь
module_wrapper/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namemodule_wrapper/conv2d/kernel
Х
0module_wrapper/conv2d/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper/conv2d/kernel*&
_output_shapes
:@*
dtype0
М
module_wrapper/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namemodule_wrapper/conv2d/bias
Е
.module_wrapper/conv2d/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper/conv2d/bias*
_output_shapes
:@*
dtype0
д
 module_wrapper_3/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*1
shared_name" module_wrapper_3/conv2d_1/kernel
Э
4module_wrapper_3/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_3/conv2d_1/kernel*&
_output_shapes
:@@*
dtype0
Ф
module_wrapper_3/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name module_wrapper_3/conv2d_1/bias
Н
2module_wrapper_3/conv2d_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_3/conv2d_1/bias*
_output_shapes
:@*
dtype0
е
 module_wrapper_6/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*1
shared_name" module_wrapper_6/conv2d_2/kernel
Ю
4module_wrapper_6/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_6/conv2d_2/kernel*'
_output_shapes
:@А*
dtype0
Х
module_wrapper_6/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name module_wrapper_6/conv2d_2/bias
О
2module_wrapper_6/conv2d_2/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_6/conv2d_2/bias*
_output_shapes	
:А*
dtype0
Щ
module_wrapper_10/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А$@*/
shared_name module_wrapper_10/dense/kernel
Т
2module_wrapper_10/dense/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_10/dense/kernel*
_output_shapes
:	А$@*
dtype0
Р
module_wrapper_10/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namemodule_wrapper_10/dense/bias
Й
0module_wrapper_10/dense/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_10/dense/bias*
_output_shapes
:@*
dtype0
Ь
 module_wrapper_12/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *1
shared_name" module_wrapper_12/dense_1/kernel
Х
4module_wrapper_12/dense_1/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_12/dense_1/kernel*
_output_shapes

:@ *
dtype0
Ф
module_wrapper_12/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name module_wrapper_12/dense_1/bias
Н
2module_wrapper_12/dense_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_12/dense_1/bias*
_output_shapes
: *
dtype0
Ь
 module_wrapper_14/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" module_wrapper_14/dense_2/kernel
Х
4module_wrapper_14/dense_2/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_14/dense_2/kernel*
_output_shapes

: *
dtype0
Ф
module_wrapper_14/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name module_wrapper_14/dense_2/bias
Н
2module_wrapper_14/dense_2/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_14/dense_2/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
о
%Adamax/module_wrapper/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adamax/module_wrapper/conv2d/kernel/m
з
9Adamax/module_wrapper/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp%Adamax/module_wrapper/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
Ю
#Adamax/module_wrapper/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adamax/module_wrapper/conv2d/bias/m
Ч
7Adamax/module_wrapper/conv2d/bias/m/Read/ReadVariableOpReadVariableOp#Adamax/module_wrapper/conv2d/bias/m*
_output_shapes
:@*
dtype0
╢
)Adamax/module_wrapper_3/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)Adamax/module_wrapper_3/conv2d_1/kernel/m
п
=Adamax/module_wrapper_3/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp)Adamax/module_wrapper_3/conv2d_1/kernel/m*&
_output_shapes
:@@*
dtype0
ж
'Adamax/module_wrapper_3/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adamax/module_wrapper_3/conv2d_1/bias/m
Я
;Adamax/module_wrapper_3/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_3/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
╖
)Adamax/module_wrapper_6/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*:
shared_name+)Adamax/module_wrapper_6/conv2d_2/kernel/m
░
=Adamax/module_wrapper_6/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp)Adamax/module_wrapper_6/conv2d_2/kernel/m*'
_output_shapes
:@А*
dtype0
з
'Adamax/module_wrapper_6/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'Adamax/module_wrapper_6/conv2d_2/bias/m
а
;Adamax/module_wrapper_6/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_6/conv2d_2/bias/m*
_output_shapes	
:А*
dtype0
л
'Adamax/module_wrapper_10/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А$@*8
shared_name)'Adamax/module_wrapper_10/dense/kernel/m
д
;Adamax/module_wrapper_10/dense/kernel/m/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_10/dense/kernel/m*
_output_shapes
:	А$@*
dtype0
в
%Adamax/module_wrapper_10/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adamax/module_wrapper_10/dense/bias/m
Ы
9Adamax/module_wrapper_10/dense/bias/m/Read/ReadVariableOpReadVariableOp%Adamax/module_wrapper_10/dense/bias/m*
_output_shapes
:@*
dtype0
о
)Adamax/module_wrapper_12/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *:
shared_name+)Adamax/module_wrapper_12/dense_1/kernel/m
з
=Adamax/module_wrapper_12/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp)Adamax/module_wrapper_12/dense_1/kernel/m*
_output_shapes

:@ *
dtype0
ж
'Adamax/module_wrapper_12/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adamax/module_wrapper_12/dense_1/bias/m
Я
;Adamax/module_wrapper_12/dense_1/bias/m/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_12/dense_1/bias/m*
_output_shapes
: *
dtype0
о
)Adamax/module_wrapper_14/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)Adamax/module_wrapper_14/dense_2/kernel/m
з
=Adamax/module_wrapper_14/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp)Adamax/module_wrapper_14/dense_2/kernel/m*
_output_shapes

: *
dtype0
ж
'Adamax/module_wrapper_14/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adamax/module_wrapper_14/dense_2/bias/m
Я
;Adamax/module_wrapper_14/dense_2/bias/m/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_14/dense_2/bias/m*
_output_shapes
:*
dtype0
о
%Adamax/module_wrapper/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adamax/module_wrapper/conv2d/kernel/v
з
9Adamax/module_wrapper/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp%Adamax/module_wrapper/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
Ю
#Adamax/module_wrapper/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adamax/module_wrapper/conv2d/bias/v
Ч
7Adamax/module_wrapper/conv2d/bias/v/Read/ReadVariableOpReadVariableOp#Adamax/module_wrapper/conv2d/bias/v*
_output_shapes
:@*
dtype0
╢
)Adamax/module_wrapper_3/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)Adamax/module_wrapper_3/conv2d_1/kernel/v
п
=Adamax/module_wrapper_3/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp)Adamax/module_wrapper_3/conv2d_1/kernel/v*&
_output_shapes
:@@*
dtype0
ж
'Adamax/module_wrapper_3/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adamax/module_wrapper_3/conv2d_1/bias/v
Я
;Adamax/module_wrapper_3/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_3/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
╖
)Adamax/module_wrapper_6/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*:
shared_name+)Adamax/module_wrapper_6/conv2d_2/kernel/v
░
=Adamax/module_wrapper_6/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp)Adamax/module_wrapper_6/conv2d_2/kernel/v*'
_output_shapes
:@А*
dtype0
з
'Adamax/module_wrapper_6/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'Adamax/module_wrapper_6/conv2d_2/bias/v
а
;Adamax/module_wrapper_6/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_6/conv2d_2/bias/v*
_output_shapes	
:А*
dtype0
л
'Adamax/module_wrapper_10/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А$@*8
shared_name)'Adamax/module_wrapper_10/dense/kernel/v
д
;Adamax/module_wrapper_10/dense/kernel/v/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_10/dense/kernel/v*
_output_shapes
:	А$@*
dtype0
в
%Adamax/module_wrapper_10/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adamax/module_wrapper_10/dense/bias/v
Ы
9Adamax/module_wrapper_10/dense/bias/v/Read/ReadVariableOpReadVariableOp%Adamax/module_wrapper_10/dense/bias/v*
_output_shapes
:@*
dtype0
о
)Adamax/module_wrapper_12/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *:
shared_name+)Adamax/module_wrapper_12/dense_1/kernel/v
з
=Adamax/module_wrapper_12/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp)Adamax/module_wrapper_12/dense_1/kernel/v*
_output_shapes

:@ *
dtype0
ж
'Adamax/module_wrapper_12/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adamax/module_wrapper_12/dense_1/bias/v
Я
;Adamax/module_wrapper_12/dense_1/bias/v/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_12/dense_1/bias/v*
_output_shapes
: *
dtype0
о
)Adamax/module_wrapper_14/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)Adamax/module_wrapper_14/dense_2/kernel/v
з
=Adamax/module_wrapper_14/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp)Adamax/module_wrapper_14/dense_2/kernel/v*
_output_shapes

: *
dtype0
ж
'Adamax/module_wrapper_14/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adamax/module_wrapper_14/dense_2/bias/v
Я
;Adamax/module_wrapper_14/dense_2/bias/v/Read/ReadVariableOpReadVariableOp'Adamax/module_wrapper_14/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ч}
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*в}
valueШ}BХ} BО}
╒
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
layer-13
layer_with_weights-5
layer-14
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
_
_module
	variables
trainable_variables
regularization_losses
	keras_api
_
_module
	variables
trainable_variables
regularization_losses
	keras_api
_
 _module
!	variables
"trainable_variables
#regularization_losses
$	keras_api
_
%_module
&	variables
'trainable_variables
(regularization_losses
)	keras_api
_
*_module
+	variables
,trainable_variables
-regularization_losses
.	keras_api
_
/_module
0	variables
1trainable_variables
2regularization_losses
3	keras_api
_
4_module
5	variables
6trainable_variables
7regularization_losses
8	keras_api
_
9_module
:	variables
;trainable_variables
<regularization_losses
=	keras_api
_
>_module
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
_
C_module
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
_
H_module
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
_
M_module
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
_
R_module
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
_
W_module
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
_
\_module
]	variables
^trainable_variables
_regularization_losses
`	keras_api
░
aiter

bbeta_1

cbeta_2
	ddecay
elearning_ratefm╘gm╒hm╓im╫jm╪km┘lm┌mm█nm▄om▌pm▐qm▀fvрgvсhvтivуjvфkvхlvцmvчnvшovщpvъqvы
V
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
 
V
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
н
rnon_trainable_variables

slayers
trainable_variables
regularization_losses
tlayer_metrics
umetrics
	variables
vlayer_regularization_losses
 
h

fkernel
gbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api

f0
g1

f0
g1
 
н
	variables
{non_trainable_variables
|layer_metrics
trainable_variables
regularization_losses
}metrics

~layers
layer_regularization_losses
V
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
 
 
 
▓
	variables
Дnon_trainable_variables
Еlayer_metrics
trainable_variables
regularization_losses
Жmetrics
Зlayers
 Иlayer_regularization_losses
V
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
 
 
 
▓
!	variables
Нnon_trainable_variables
Оlayer_metrics
"trainable_variables
#regularization_losses
Пmetrics
Рlayers
 Сlayer_regularization_losses
l

hkernel
ibias
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api

h0
i1

h0
i1
 
▓
&	variables
Цnon_trainable_variables
Чlayer_metrics
'trainable_variables
(regularization_losses
Шmetrics
Щlayers
 Ъlayer_regularization_losses
V
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
 
 
 
▓
+	variables
Яnon_trainable_variables
аlayer_metrics
,trainable_variables
-regularization_losses
бmetrics
вlayers
 гlayer_regularization_losses
V
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
 
 
 
▓
0	variables
иnon_trainable_variables
йlayer_metrics
1trainable_variables
2regularization_losses
кmetrics
лlayers
 мlayer_regularization_losses
l

jkernel
kbias
н	variables
оtrainable_variables
пregularization_losses
░	keras_api

j0
k1

j0
k1
 
▓
5	variables
▒non_trainable_variables
▓layer_metrics
6trainable_variables
7regularization_losses
│metrics
┤layers
 ╡layer_regularization_losses
V
╢	variables
╖trainable_variables
╕regularization_losses
╣	keras_api
 
 
 
▓
:	variables
║non_trainable_variables
╗layer_metrics
;trainable_variables
<regularization_losses
╝metrics
╜layers
 ╛layer_regularization_losses
V
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
 
 
 
▓
?	variables
├non_trainable_variables
─layer_metrics
@trainable_variables
Aregularization_losses
┼metrics
╞layers
 ╟layer_regularization_losses
V
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
 
 
 
▓
D	variables
╠non_trainable_variables
═layer_metrics
Etrainable_variables
Fregularization_losses
╬metrics
╧layers
 ╨layer_regularization_losses
l

lkernel
mbias
╤	variables
╥trainable_variables
╙regularization_losses
╘	keras_api

l0
m1

l0
m1
 
▓
I	variables
╒non_trainable_variables
╓layer_metrics
Jtrainable_variables
Kregularization_losses
╫metrics
╪layers
 ┘layer_regularization_losses
V
┌	variables
█trainable_variables
▄regularization_losses
▌	keras_api
 
 
 
▓
N	variables
▐non_trainable_variables
▀layer_metrics
Otrainable_variables
Pregularization_losses
рmetrics
сlayers
 тlayer_regularization_losses
l

nkernel
obias
у	variables
фtrainable_variables
хregularization_losses
ц	keras_api

n0
o1

n0
o1
 
▓
S	variables
чnon_trainable_variables
шlayer_metrics
Ttrainable_variables
Uregularization_losses
щmetrics
ъlayers
 ыlayer_regularization_losses
V
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
 
 
 
▓
X	variables
Ёnon_trainable_variables
ёlayer_metrics
Ytrainable_variables
Zregularization_losses
Єmetrics
єlayers
 Їlayer_regularization_losses
l

pkernel
qbias
ї	variables
Ўtrainable_variables
ўregularization_losses
°	keras_api

p0
q1

p0
q1
 
▓
]	variables
∙non_trainable_variables
·layer_metrics
^trainable_variables
_regularization_losses
√metrics
№layers
 ¤layer_regularization_losses
JH
VARIABLE_VALUEAdamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdamax/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdamax/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdamax/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEAdamax/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEmodule_wrapper/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEmodule_wrapper/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE module_wrapper_3/conv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodule_wrapper_3/conv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE module_wrapper_6/conv2d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodule_wrapper_6/conv2d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodule_wrapper_10/dense/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEmodule_wrapper_10/dense/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE module_wrapper_12/dense_1/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmodule_wrapper_12/dense_1/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE module_wrapper_14/dense_2/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEmodule_wrapper_14/dense_2/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
n
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
 

■0
 1
 

f0
g1

f0
g1
 
▓
w	variables
Аnon_trainable_variables
Бlayer_metrics
xtrainable_variables
yregularization_losses
Вmetrics
Гlayers
 Дlayer_regularization_losses
 
 
 
 
 
 
 
 
╡
А	variables
Еnon_trainable_variables
Жlayer_metrics
Бtrainable_variables
Вregularization_losses
Зmetrics
Иlayers
 Йlayer_regularization_losses
 
 
 
 
 
 
 
 
╡
Й	variables
Кnon_trainable_variables
Лlayer_metrics
Кtrainable_variables
Лregularization_losses
Мmetrics
Нlayers
 Оlayer_regularization_losses
 
 
 
 
 

h0
i1

h0
i1
 
╡
Т	variables
Пnon_trainable_variables
Рlayer_metrics
Уtrainable_variables
Фregularization_losses
Сmetrics
Тlayers
 Уlayer_regularization_losses
 
 
 
 
 
 
 
 
╡
Ы	variables
Фnon_trainable_variables
Хlayer_metrics
Ьtrainable_variables
Эregularization_losses
Цmetrics
Чlayers
 Шlayer_regularization_losses
 
 
 
 
 
 
 
 
╡
д	variables
Щnon_trainable_variables
Ъlayer_metrics
еtrainable_variables
жregularization_losses
Ыmetrics
Ьlayers
 Эlayer_regularization_losses
 
 
 
 
 

j0
k1

j0
k1
 
╡
н	variables
Юnon_trainable_variables
Яlayer_metrics
оtrainable_variables
пregularization_losses
аmetrics
бlayers
 вlayer_regularization_losses
 
 
 
 
 
 
 
 
╡
╢	variables
гnon_trainable_variables
дlayer_metrics
╖trainable_variables
╕regularization_losses
еmetrics
жlayers
 зlayer_regularization_losses
 
 
 
 
 
 
 
 
╡
┐	variables
иnon_trainable_variables
йlayer_metrics
└trainable_variables
┴regularization_losses
кmetrics
лlayers
 мlayer_regularization_losses
 
 
 
 
 
 
 
 
╡
╚	variables
нnon_trainable_variables
оlayer_metrics
╔trainable_variables
╩regularization_losses
пmetrics
░layers
 ▒layer_regularization_losses
 
 
 
 
 

l0
m1

l0
m1
 
╡
╤	variables
▓non_trainable_variables
│layer_metrics
╥trainable_variables
╙regularization_losses
┤metrics
╡layers
 ╢layer_regularization_losses
 
 
 
 
 
 
 
 
╡
┌	variables
╖non_trainable_variables
╕layer_metrics
█trainable_variables
▄regularization_losses
╣metrics
║layers
 ╗layer_regularization_losses
 
 
 
 
 

n0
o1

n0
o1
 
╡
у	variables
╝non_trainable_variables
╜layer_metrics
фtrainable_variables
хregularization_losses
╛metrics
┐layers
 └layer_regularization_losses
 
 
 
 
 
 
 
 
╡
ь	variables
┴non_trainable_variables
┬layer_metrics
эtrainable_variables
юregularization_losses
├metrics
─layers
 ┼layer_regularization_losses
 
 
 
 
 

p0
q1

p0
q1
 
╡
ї	variables
╞non_trainable_variables
╟layer_metrics
Ўtrainable_variables
ўregularization_losses
╚metrics
╔layers
 ╩layer_regularization_losses
 
 
 
 
 
8

╦total

╠count
═	variables
╬	keras_api
I

╧total

╨count
╤
_fn_kwargs
╥	variables
╙	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

╦0
╠1

═	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

╧0
╨1

╥	variables
ИЕ
VARIABLE_VALUE%Adamax/module_wrapper/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE#Adamax/module_wrapper/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)Adamax/module_wrapper_3/conv2d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'Adamax/module_wrapper_3/conv2d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)Adamax/module_wrapper_6/conv2d_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'Adamax/module_wrapper_6/conv2d_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'Adamax/module_wrapper_10/dense/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%Adamax/module_wrapper_10/dense/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)Adamax/module_wrapper_12/dense_1/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'Adamax/module_wrapper_12/dense_1/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)Adamax/module_wrapper_14/dense_2/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'Adamax/module_wrapper_14/dense_2/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%Adamax/module_wrapper/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUE#Adamax/module_wrapper/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)Adamax/module_wrapper_3/conv2d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'Adamax/module_wrapper_3/conv2d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)Adamax/module_wrapper_6/conv2d_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'Adamax/module_wrapper_6/conv2d_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'Adamax/module_wrapper_10/dense/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%Adamax/module_wrapper_10/dense/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)Adamax/module_wrapper_12/dense_1/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'Adamax/module_wrapper_12/dense_1/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)Adamax/module_wrapper_14/dense_2/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'Adamax/module_wrapper_14/dense_2/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ч
$serving_default_module_wrapper_inputPlaceholder*/
_output_shapes
:         00*
dtype0*$
shape:         00
┌
StatefulPartitionedCallStatefulPartitionedCall$serving_default_module_wrapper_inputmodule_wrapper/conv2d/kernelmodule_wrapper/conv2d/bias module_wrapper_3/conv2d_1/kernelmodule_wrapper_3/conv2d_1/bias module_wrapper_6/conv2d_2/kernelmodule_wrapper_6/conv2d_2/biasmodule_wrapper_10/dense/kernelmodule_wrapper_10/dense/bias module_wrapper_12/dense_1/kernelmodule_wrapper_12/dense_1/bias module_wrapper_14/dense_2/kernelmodule_wrapper_14/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_388509
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdamax/iter/Read/ReadVariableOp!Adamax/beta_1/Read/ReadVariableOp!Adamax/beta_2/Read/ReadVariableOp Adamax/decay/Read/ReadVariableOp(Adamax/learning_rate/Read/ReadVariableOp0module_wrapper/conv2d/kernel/Read/ReadVariableOp.module_wrapper/conv2d/bias/Read/ReadVariableOp4module_wrapper_3/conv2d_1/kernel/Read/ReadVariableOp2module_wrapper_3/conv2d_1/bias/Read/ReadVariableOp4module_wrapper_6/conv2d_2/kernel/Read/ReadVariableOp2module_wrapper_6/conv2d_2/bias/Read/ReadVariableOp2module_wrapper_10/dense/kernel/Read/ReadVariableOp0module_wrapper_10/dense/bias/Read/ReadVariableOp4module_wrapper_12/dense_1/kernel/Read/ReadVariableOp2module_wrapper_12/dense_1/bias/Read/ReadVariableOp4module_wrapper_14/dense_2/kernel/Read/ReadVariableOp2module_wrapper_14/dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp9Adamax/module_wrapper/conv2d/kernel/m/Read/ReadVariableOp7Adamax/module_wrapper/conv2d/bias/m/Read/ReadVariableOp=Adamax/module_wrapper_3/conv2d_1/kernel/m/Read/ReadVariableOp;Adamax/module_wrapper_3/conv2d_1/bias/m/Read/ReadVariableOp=Adamax/module_wrapper_6/conv2d_2/kernel/m/Read/ReadVariableOp;Adamax/module_wrapper_6/conv2d_2/bias/m/Read/ReadVariableOp;Adamax/module_wrapper_10/dense/kernel/m/Read/ReadVariableOp9Adamax/module_wrapper_10/dense/bias/m/Read/ReadVariableOp=Adamax/module_wrapper_12/dense_1/kernel/m/Read/ReadVariableOp;Adamax/module_wrapper_12/dense_1/bias/m/Read/ReadVariableOp=Adamax/module_wrapper_14/dense_2/kernel/m/Read/ReadVariableOp;Adamax/module_wrapper_14/dense_2/bias/m/Read/ReadVariableOp9Adamax/module_wrapper/conv2d/kernel/v/Read/ReadVariableOp7Adamax/module_wrapper/conv2d/bias/v/Read/ReadVariableOp=Adamax/module_wrapper_3/conv2d_1/kernel/v/Read/ReadVariableOp;Adamax/module_wrapper_3/conv2d_1/bias/v/Read/ReadVariableOp=Adamax/module_wrapper_6/conv2d_2/kernel/v/Read/ReadVariableOp;Adamax/module_wrapper_6/conv2d_2/bias/v/Read/ReadVariableOp;Adamax/module_wrapper_10/dense/kernel/v/Read/ReadVariableOp9Adamax/module_wrapper_10/dense/bias/v/Read/ReadVariableOp=Adamax/module_wrapper_12/dense_1/kernel/v/Read/ReadVariableOp;Adamax/module_wrapper_12/dense_1/bias/v/Read/ReadVariableOp=Adamax/module_wrapper_14/dense_2/kernel/v/Read/ReadVariableOp;Adamax/module_wrapper_14/dense_2/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_389366
Н
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameAdamax/iterAdamax/beta_1Adamax/beta_2Adamax/decayAdamax/learning_ratemodule_wrapper/conv2d/kernelmodule_wrapper/conv2d/bias module_wrapper_3/conv2d_1/kernelmodule_wrapper_3/conv2d_1/bias module_wrapper_6/conv2d_2/kernelmodule_wrapper_6/conv2d_2/biasmodule_wrapper_10/dense/kernelmodule_wrapper_10/dense/bias module_wrapper_12/dense_1/kernelmodule_wrapper_12/dense_1/bias module_wrapper_14/dense_2/kernelmodule_wrapper_14/dense_2/biastotalcounttotal_1count_1%Adamax/module_wrapper/conv2d/kernel/m#Adamax/module_wrapper/conv2d/bias/m)Adamax/module_wrapper_3/conv2d_1/kernel/m'Adamax/module_wrapper_3/conv2d_1/bias/m)Adamax/module_wrapper_6/conv2d_2/kernel/m'Adamax/module_wrapper_6/conv2d_2/bias/m'Adamax/module_wrapper_10/dense/kernel/m%Adamax/module_wrapper_10/dense/bias/m)Adamax/module_wrapper_12/dense_1/kernel/m'Adamax/module_wrapper_12/dense_1/bias/m)Adamax/module_wrapper_14/dense_2/kernel/m'Adamax/module_wrapper_14/dense_2/bias/m%Adamax/module_wrapper/conv2d/kernel/v#Adamax/module_wrapper/conv2d/bias/v)Adamax/module_wrapper_3/conv2d_1/kernel/v'Adamax/module_wrapper_3/conv2d_1/bias/v)Adamax/module_wrapper_6/conv2d_2/kernel/v'Adamax/module_wrapper_6/conv2d_2/bias/v'Adamax/module_wrapper_10/dense/kernel/v%Adamax/module_wrapper_10/dense/bias/v)Adamax/module_wrapper_12/dense_1/kernel/v'Adamax/module_wrapper_12/dense_1/bias/v)Adamax/module_wrapper_14/dense_2/kernel/v'Adamax/module_wrapper_14/dense_2/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_389511┘Щ
б
h
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_387801

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstА
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         А$2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
│
а
2__inference_module_wrapper_10_layer_call_fn_389065

args_0
unknown:	А$@
	unknown_0:@
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_3878142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А$
 
_user_specified_nameargs_0
Аf
ж
__inference__traced_save_389366
file_prefix*
&savev2_adamax_iter_read_readvariableop	,
(savev2_adamax_beta_1_read_readvariableop,
(savev2_adamax_beta_2_read_readvariableop+
'savev2_adamax_decay_read_readvariableop3
/savev2_adamax_learning_rate_read_readvariableop;
7savev2_module_wrapper_conv2d_kernel_read_readvariableop9
5savev2_module_wrapper_conv2d_bias_read_readvariableop?
;savev2_module_wrapper_3_conv2d_1_kernel_read_readvariableop=
9savev2_module_wrapper_3_conv2d_1_bias_read_readvariableop?
;savev2_module_wrapper_6_conv2d_2_kernel_read_readvariableop=
9savev2_module_wrapper_6_conv2d_2_bias_read_readvariableop=
9savev2_module_wrapper_10_dense_kernel_read_readvariableop;
7savev2_module_wrapper_10_dense_bias_read_readvariableop?
;savev2_module_wrapper_12_dense_1_kernel_read_readvariableop=
9savev2_module_wrapper_12_dense_1_bias_read_readvariableop?
;savev2_module_wrapper_14_dense_2_kernel_read_readvariableop=
9savev2_module_wrapper_14_dense_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopD
@savev2_adamax_module_wrapper_conv2d_kernel_m_read_readvariableopB
>savev2_adamax_module_wrapper_conv2d_bias_m_read_readvariableopH
Dsavev2_adamax_module_wrapper_3_conv2d_1_kernel_m_read_readvariableopF
Bsavev2_adamax_module_wrapper_3_conv2d_1_bias_m_read_readvariableopH
Dsavev2_adamax_module_wrapper_6_conv2d_2_kernel_m_read_readvariableopF
Bsavev2_adamax_module_wrapper_6_conv2d_2_bias_m_read_readvariableopF
Bsavev2_adamax_module_wrapper_10_dense_kernel_m_read_readvariableopD
@savev2_adamax_module_wrapper_10_dense_bias_m_read_readvariableopH
Dsavev2_adamax_module_wrapper_12_dense_1_kernel_m_read_readvariableopF
Bsavev2_adamax_module_wrapper_12_dense_1_bias_m_read_readvariableopH
Dsavev2_adamax_module_wrapper_14_dense_2_kernel_m_read_readvariableopF
Bsavev2_adamax_module_wrapper_14_dense_2_bias_m_read_readvariableopD
@savev2_adamax_module_wrapper_conv2d_kernel_v_read_readvariableopB
>savev2_adamax_module_wrapper_conv2d_bias_v_read_readvariableopH
Dsavev2_adamax_module_wrapper_3_conv2d_1_kernel_v_read_readvariableopF
Bsavev2_adamax_module_wrapper_3_conv2d_1_bias_v_read_readvariableopH
Dsavev2_adamax_module_wrapper_6_conv2d_2_kernel_v_read_readvariableopF
Bsavev2_adamax_module_wrapper_6_conv2d_2_bias_v_read_readvariableopF
Bsavev2_adamax_module_wrapper_10_dense_kernel_v_read_readvariableopD
@savev2_adamax_module_wrapper_10_dense_bias_v_read_readvariableopH
Dsavev2_adamax_module_wrapper_12_dense_1_kernel_v_read_readvariableopF
Bsavev2_adamax_module_wrapper_12_dense_1_bias_v_read_readvariableopH
Dsavev2_adamax_module_wrapper_14_dense_2_kernel_v_read_readvariableopF
Bsavev2_adamax_module_wrapper_14_dense_2_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameМ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ю
valueФBС.B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesэ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_adamax_iter_read_readvariableop(savev2_adamax_beta_1_read_readvariableop(savev2_adamax_beta_2_read_readvariableop'savev2_adamax_decay_read_readvariableop/savev2_adamax_learning_rate_read_readvariableop7savev2_module_wrapper_conv2d_kernel_read_readvariableop5savev2_module_wrapper_conv2d_bias_read_readvariableop;savev2_module_wrapper_3_conv2d_1_kernel_read_readvariableop9savev2_module_wrapper_3_conv2d_1_bias_read_readvariableop;savev2_module_wrapper_6_conv2d_2_kernel_read_readvariableop9savev2_module_wrapper_6_conv2d_2_bias_read_readvariableop9savev2_module_wrapper_10_dense_kernel_read_readvariableop7savev2_module_wrapper_10_dense_bias_read_readvariableop;savev2_module_wrapper_12_dense_1_kernel_read_readvariableop9savev2_module_wrapper_12_dense_1_bias_read_readvariableop;savev2_module_wrapper_14_dense_2_kernel_read_readvariableop9savev2_module_wrapper_14_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop@savev2_adamax_module_wrapper_conv2d_kernel_m_read_readvariableop>savev2_adamax_module_wrapper_conv2d_bias_m_read_readvariableopDsavev2_adamax_module_wrapper_3_conv2d_1_kernel_m_read_readvariableopBsavev2_adamax_module_wrapper_3_conv2d_1_bias_m_read_readvariableopDsavev2_adamax_module_wrapper_6_conv2d_2_kernel_m_read_readvariableopBsavev2_adamax_module_wrapper_6_conv2d_2_bias_m_read_readvariableopBsavev2_adamax_module_wrapper_10_dense_kernel_m_read_readvariableop@savev2_adamax_module_wrapper_10_dense_bias_m_read_readvariableopDsavev2_adamax_module_wrapper_12_dense_1_kernel_m_read_readvariableopBsavev2_adamax_module_wrapper_12_dense_1_bias_m_read_readvariableopDsavev2_adamax_module_wrapper_14_dense_2_kernel_m_read_readvariableopBsavev2_adamax_module_wrapper_14_dense_2_bias_m_read_readvariableop@savev2_adamax_module_wrapper_conv2d_kernel_v_read_readvariableop>savev2_adamax_module_wrapper_conv2d_bias_v_read_readvariableopDsavev2_adamax_module_wrapper_3_conv2d_1_kernel_v_read_readvariableopBsavev2_adamax_module_wrapper_3_conv2d_1_bias_v_read_readvariableopDsavev2_adamax_module_wrapper_6_conv2d_2_kernel_v_read_readvariableopBsavev2_adamax_module_wrapper_6_conv2d_2_bias_v_read_readvariableopBsavev2_adamax_module_wrapper_10_dense_kernel_v_read_readvariableop@savev2_adamax_module_wrapper_10_dense_bias_v_read_readvariableopDsavev2_adamax_module_wrapper_12_dense_1_kernel_v_read_readvariableopBsavev2_adamax_module_wrapper_12_dense_1_bias_v_read_readvariableopDsavev2_adamax_module_wrapper_14_dense_2_kernel_v_read_readvariableopBsavev2_adamax_module_wrapper_14_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ь
_input_shapesК
З: : : : : : :@:@:@@:@:@А:А:	А$@:@:@ : : :: : : : :@:@:@@:@:@А:А:	А$@:@:@ : : ::@:@:@@:@:@А:А:	А$@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 	

_output_shapes
:@:-
)
'
_output_shapes
:@А:!

_output_shapes	
:А:%!

_output_shapes
:	А$@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:%!

_output_shapes
:	А$@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$  

_output_shapes

: : !

_output_shapes
::,"(
&
_output_shapes
:@: #

_output_shapes
:@:,$(
&
_output_shapes
:@@: %

_output_shapes
:@:-&)
'
_output_shapes
:@А:!'

_output_shapes	
:А:%(!

_output_shapes
:	А$@: )

_output_shapes
:@:$* 

_output_shapes

:@ : +

_output_shapes
: :$, 

_output_shapes

: : -

_output_shapes
::.

_output_shapes
: 
Ц
Ю
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_387862

args_08
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:
identityИвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpе
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOpЛ
dense_2/MatMulMatMulargs_0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2/Softmaxо
IdentityIdentitydense_2/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
є
h
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_388085

args_0
identity│
max_pooling2d_2/MaxPoolMaxPoolargs_0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool}
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
у
M
1__inference_module_wrapper_9_layer_call_fn_389034

args_0
identity╬
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_3880462
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
░
Я
2__inference_module_wrapper_12_layer_call_fn_389132

args_0
unknown:@ 
	unknown_0: 
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_3878382
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
щ
h
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_387724

args_0
identityо
max_pooling2d/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolz
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         00@:W S
/
_output_shapes
:         00@
 
_user_specified_nameargs_0
я
h
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_388154

args_0
identity▓
max_pooling2d_1/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
░
Я
2__inference_module_wrapper_14_layer_call_fn_389199

args_0
unknown: 
	unknown_0:
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_3878622
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
▄
Ч
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_389045

args_07
$dense_matmul_readvariableop_resource:	А$@3
%dense_biasadd_readvariableop_resource:@
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А$@*
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2

dense/Reluй
IdentityIdentitydense/Relu:activations:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А$: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А$
 
_user_specified_nameargs_0
з
h
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_387731

args_0
identityr
dropout/IdentityIdentityargs_0*
T0*/
_output_shapes
:         @2
dropout/Identityu
IdentityIdentitydropout/Identity:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ч
л
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_387775

args_0B
'conv2d_2_conv2d_readvariableop_resource:@А7
(conv2d_2_biasadd_readvariableop_resource:	А
identityИвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOp▒
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_2/Conv2D/ReadVariableOp┐
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_2/Conv2Dи
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpн
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_2/Relu╗
IdentityIdentityconv2d_2/Relu:activations:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
■
k
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_388069

args_0
identityИw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/ConstЪ
dropout_2/dropout/MulMulargs_0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout_2/dropout/Mulh
dropout_2/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeы
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype0*
seedО╠ЮЇ20
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yя
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2 
dropout_2/dropout/GreaterEqualж
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_2/dropout/Castл
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_2/dropout/Mul_1x
IdentityIdentitydropout_2/dropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╖
l
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_387999

args_0
identityИw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_3/dropout/ConstС
dropout_3/dropout/MulMulargs_0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout_3/dropout/Mulh
dropout_3/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeт
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ20
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2"
 dropout_3/dropout/GreaterEqual/yц
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2 
dropout_3/dropout/GreaterEqualЭ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout_3/dropout/Castв
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout_3/dropout/Mul_1o
IdentityIdentitydropout_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
з
h
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_388816

args_0
identityr
dropout/IdentityIdentityargs_0*
T0*/
_output_shapes
:         @2
dropout/Identityu
IdentityIdentitydropout/Identity:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
╥
д
/__inference_module_wrapper_layer_call_fn_388782

args_0!
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_3877132
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         00
 
_user_specified_nameargs_0
┌
L
0__inference_max_pooling2d_1_layer_call_fn_388534

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3885282
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
П
й
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_388180

args_0A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@
identityИвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╛
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Relu║
IdentityIdentityconv2d_1/Relu:activations:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ц
Ю
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_389179

args_08
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:
identityИвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpе
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOpЛ
dense_2/MatMulMatMulargs_0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2/Softmaxо
IdentityIdentitydense_2/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
О
Ю
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_387972

args_08
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOpЛ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/Reluп
IdentityIdentitydense_1/Relu:activations:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
є
M
1__inference_module_wrapper_7_layer_call_fn_388985

args_0
identity╓
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_3880852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╫
Я
J__inference_module_wrapper_layer_call_and_return_conditional_losses_388249

args_0?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╕
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@*
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         00@2
conv2d/Relu┤
IdentityIdentityconv2d/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         00
 
_user_specified_nameargs_0
▒
h
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_387793

args_0
identityw
dropout_2/IdentityIdentityargs_0*
T0*0
_output_shapes
:         А2
dropout_2/Identityx
IdentityIdentitydropout_2/Identity:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╖
l
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_387946

args_0
identityИw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_4/dropout/ConstС
dropout_4/dropout/MulMulargs_0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout_4/dropout/Mulh
dropout_4/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_4/dropout/Shapeт
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seedО╠ЮЇ20
.dropout_4/dropout/random_uniform/RandomUniformЙ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2"
 dropout_4/dropout/GreaterEqual/yц
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2 
dropout_4/dropout/GreaterEqualЭ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_4/dropout/Castв
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_4/dropout/Mul_1o
IdentityIdentitydropout_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
я
M
1__inference_module_wrapper_4_layer_call_fn_388898

args_0
identity╒
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_3881542
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
▄
Ч
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_388025

args_07
$dense_matmul_readvariableop_resource:	А$@3
%dense_biasadd_readvariableop_resource:@
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А$@*
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2

dense/Reluй
IdentityIdentitydense/Relu:activations:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А$: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А$
 
_user_specified_nameargs_0
▌
k
2__inference_module_wrapper_11_layer_call_fn_389101

args_0
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_3879992
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
ё

╔
$__inference_signature_wrapper_388509
module_wrapper_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А$@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_3876952
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:         00
.
_user_specified_namemodule_wrapper_input
Э
╨
+__inference_sequential_layer_call_fn_387896
module_wrapper_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А$@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3878692
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:         00
.
_user_specified_namemodule_wrapper_input
√
j
1__inference_module_wrapper_2_layer_call_fn_388838

args_0
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_3882072
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
є
h
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_388975

args_0
identity│
max_pooling2d_2/MaxPoolMaxPoolargs_0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool}
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
є
h
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_388970

args_0
identity│
max_pooling2d_2/MaxPoolMaxPoolargs_0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool}
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
О
Ю
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_389112

args_08
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOpЛ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/Reluп
IdentityIdentitydense_1/Relu:activations:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
є
h
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_387786

args_0
identity│
max_pooling2d_2/MaxPoolMaxPoolargs_0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool}
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
Ц
Ю
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_387919

args_08
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:
identityИвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpе
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOpЛ
dense_2/MatMulMatMulargs_0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2/Softmaxо
IdentityIdentitydense_2/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
я
M
1__inference_module_wrapper_1_layer_call_fn_388811

args_0
identity╒
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_3882232
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         00@:W S
/
_output_shapes
:         00@
 
_user_specified_nameargs_0
П
й
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_388849

args_0A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@
identityИвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╛
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Relu║
IdentityIdentityconv2d_1/Relu:activations:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
╫
Я
J__inference_module_wrapper_layer_call_and_return_conditional_losses_387713

args_0?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╕
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@*
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         00@2
conv2d/Relu┤
IdentityIdentityconv2d/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         00
 
_user_specified_nameargs_0
у
M
1__inference_module_wrapper_9_layer_call_fn_389029

args_0
identity╬
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_3878012
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
щC
ё
F__inference_sequential_layer_call_and_return_conditional_losses_387869

inputs/
module_wrapper_387714:@#
module_wrapper_387716:@1
module_wrapper_3_387745:@@%
module_wrapper_3_387747:@2
module_wrapper_6_387776:@А&
module_wrapper_6_387778:	А+
module_wrapper_10_387815:	А$@&
module_wrapper_10_387817:@*
module_wrapper_12_387839:@ &
module_wrapper_12_387841: *
module_wrapper_14_387863: &
module_wrapper_14_387865:
identityИв&module_wrapper/StatefulPartitionedCallв)module_wrapper_10/StatefulPartitionedCallв)module_wrapper_12/StatefulPartitionedCallв)module_wrapper_14/StatefulPartitionedCallв(module_wrapper_3/StatefulPartitionedCallв(module_wrapper_6/StatefulPartitionedCall╜
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_387714module_wrapper_387716*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_3877132(
&module_wrapper/StatefulPartitionedCallа
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_3877242"
 module_wrapper_1/PartitionedCallЪ
 module_wrapper_2/PartitionedCallPartitionedCall)module_wrapper_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_3877312"
 module_wrapper_2/PartitionedCallъ
(module_wrapper_3/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_2/PartitionedCall:output:0module_wrapper_3_387745module_wrapper_3_387747*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_3877442*
(module_wrapper_3/StatefulPartitionedCallв
 module_wrapper_4/PartitionedCallPartitionedCall1module_wrapper_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_3877552"
 module_wrapper_4/PartitionedCallЪ
 module_wrapper_5/PartitionedCallPartitionedCall)module_wrapper_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_3877622"
 module_wrapper_5/PartitionedCallы
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_387776module_wrapper_6_387778*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_3877752*
(module_wrapper_6/StatefulPartitionedCallг
 module_wrapper_7/PartitionedCallPartitionedCall1module_wrapper_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_3877862"
 module_wrapper_7/PartitionedCallЫ
 module_wrapper_8/PartitionedCallPartitionedCall)module_wrapper_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_3877932"
 module_wrapper_8/PartitionedCallУ
 module_wrapper_9/PartitionedCallPartitionedCall)module_wrapper_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_3878012"
 module_wrapper_9/PartitionedCallч
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_387815module_wrapper_10_387817*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_3878142+
)module_wrapper_10/StatefulPartitionedCallЮ
!module_wrapper_11/PartitionedCallPartitionedCall2module_wrapper_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_3878252#
!module_wrapper_11/PartitionedCallш
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_387839module_wrapper_12_387841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_3878382+
)module_wrapper_12/StatefulPartitionedCallЮ
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_3878492#
!module_wrapper_13/PartitionedCallш
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_387863module_wrapper_14_387865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_3878622+
)module_wrapper_14/StatefulPartitionedCallЙ
IdentityIdentity2module_wrapper_14/StatefulPartitionedCall:output:0'^module_wrapper/StatefulPartitionedCall*^module_wrapper_10/StatefulPartitionedCall*^module_wrapper_12/StatefulPartitionedCall*^module_wrapper_14/StatefulPartitionedCall)^module_wrapper_3/StatefulPartitionedCall)^module_wrapper_6/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2V
)module_wrapper_10/StatefulPartitionedCall)module_wrapper_10/StatefulPartitionedCall2V
)module_wrapper_12/StatefulPartitionedCall)module_wrapper_12/StatefulPartitionedCall2V
)module_wrapper_14/StatefulPartitionedCall)module_wrapper_14/StatefulPartitionedCall2T
(module_wrapper_3/StatefulPartitionedCall(module_wrapper_3/StatefulPartitionedCall2T
(module_wrapper_6/StatefulPartitionedCall(module_wrapper_6/StatefulPartitionedCall:W S
/
_output_shapes
:         00
 
_user_specified_nameinputs
┌
и
1__inference_module_wrapper_6_layer_call_fn_388965

args_0"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_3881112
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
УD
 
F__inference_sequential_layer_call_and_return_conditional_losses_388429
module_wrapper_input/
module_wrapper_388389:@#
module_wrapper_388391:@1
module_wrapper_3_388396:@@%
module_wrapper_3_388398:@2
module_wrapper_6_388403:@А&
module_wrapper_6_388405:	А+
module_wrapper_10_388411:	А$@&
module_wrapper_10_388413:@*
module_wrapper_12_388417:@ &
module_wrapper_12_388419: *
module_wrapper_14_388423: &
module_wrapper_14_388425:
identityИв&module_wrapper/StatefulPartitionedCallв)module_wrapper_10/StatefulPartitionedCallв)module_wrapper_12/StatefulPartitionedCallв)module_wrapper_14/StatefulPartitionedCallв(module_wrapper_3/StatefulPartitionedCallв(module_wrapper_6/StatefulPartitionedCall╦
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_388389module_wrapper_388391*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_3877132(
&module_wrapper/StatefulPartitionedCallа
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_3877242"
 module_wrapper_1/PartitionedCallЪ
 module_wrapper_2/PartitionedCallPartitionedCall)module_wrapper_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_3877312"
 module_wrapper_2/PartitionedCallъ
(module_wrapper_3/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_2/PartitionedCall:output:0module_wrapper_3_388396module_wrapper_3_388398*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_3877442*
(module_wrapper_3/StatefulPartitionedCallв
 module_wrapper_4/PartitionedCallPartitionedCall1module_wrapper_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_3877552"
 module_wrapper_4/PartitionedCallЪ
 module_wrapper_5/PartitionedCallPartitionedCall)module_wrapper_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_3877622"
 module_wrapper_5/PartitionedCallы
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_5/PartitionedCall:output:0module_wrapper_6_388403module_wrapper_6_388405*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_3877752*
(module_wrapper_6/StatefulPartitionedCallг
 module_wrapper_7/PartitionedCallPartitionedCall1module_wrapper_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_3877862"
 module_wrapper_7/PartitionedCallЫ
 module_wrapper_8/PartitionedCallPartitionedCall)module_wrapper_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_3877932"
 module_wrapper_8/PartitionedCallУ
 module_wrapper_9/PartitionedCallPartitionedCall)module_wrapper_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_3878012"
 module_wrapper_9/PartitionedCallч
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_388411module_wrapper_10_388413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_3878142+
)module_wrapper_10/StatefulPartitionedCallЮ
!module_wrapper_11/PartitionedCallPartitionedCall2module_wrapper_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_3878252#
!module_wrapper_11/PartitionedCallш
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_11/PartitionedCall:output:0module_wrapper_12_388417module_wrapper_12_388419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_3878382+
)module_wrapper_12/StatefulPartitionedCallЮ
!module_wrapper_13/PartitionedCallPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_3878492#
!module_wrapper_13/PartitionedCallш
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_13/PartitionedCall:output:0module_wrapper_14_388423module_wrapper_14_388425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_3878622+
)module_wrapper_14/StatefulPartitionedCallЙ
IdentityIdentity2module_wrapper_14/StatefulPartitionedCall:output:0'^module_wrapper/StatefulPartitionedCall*^module_wrapper_10/StatefulPartitionedCall*^module_wrapper_12/StatefulPartitionedCall*^module_wrapper_14/StatefulPartitionedCall)^module_wrapper_3/StatefulPartitionedCall)^module_wrapper_6/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2V
)module_wrapper_10/StatefulPartitionedCall)module_wrapper_10/StatefulPartitionedCall2V
)module_wrapper_12/StatefulPartitionedCall)module_wrapper_12/StatefulPartitionedCall2V
)module_wrapper_14/StatefulPartitionedCall)module_wrapper_14/StatefulPartitionedCall2T
(module_wrapper_3/StatefulPartitionedCall(module_wrapper_3/StatefulPartitionedCall2T
(module_wrapper_6/StatefulPartitionedCall(module_wrapper_6/StatefulPartitionedCall:e a
/
_output_shapes
:         00
.
_user_specified_namemodule_wrapper_input
╖
l
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_389091

args_0
identityИw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_3/dropout/ConstС
dropout_3/dropout/MulMulargs_0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout_3/dropout/Mulh
dropout_3/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeт
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ20
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2"
 dropout_3/dropout/GreaterEqual/yц
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2 
dropout_3/dropout/GreaterEqualЭ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout_3/dropout/Castв
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout_3/dropout/Mul_1o
IdentityIdentitydropout_3/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
й
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_388516

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
щ
h
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_388223

args_0
identityо
max_pooling2d/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolz
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         00@:W S
/
_output_shapes
:         00@
 
_user_specified_nameargs_0
░
Я
2__inference_module_wrapper_14_layer_call_fn_389208

args_0
unknown: 
	unknown_0:
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_3879192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
л
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_388540

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
н
h
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_388903

args_0
identityv
dropout_1/IdentityIdentityargs_0*
T0*/
_output_shapes
:         @2
dropout_1/Identityw
IdentityIdentitydropout_1/Identity:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
я
h
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_388883

args_0
identity▓
max_pooling2d_1/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
О
i
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_389079

args_0
identityn
dropout_3/IdentityIdentityargs_0*
T0*'
_output_shapes
:         @2
dropout_3/Identityo
IdentityIdentitydropout_3/Identity:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
┌
L
0__inference_max_pooling2d_2_layer_call_fn_388546

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_3885402
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Э
╨
+__inference_sequential_layer_call_fn_388386
module_wrapper_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А$@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3883302
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:         00
.
_user_specified_namemodule_wrapper_input
я
M
1__inference_module_wrapper_2_layer_call_fn_388833

args_0
identity╒
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_3877312
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
╤
N
2__inference_module_wrapper_13_layer_call_fn_389163

args_0
identity╬
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_3878492
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
є
M
1__inference_module_wrapper_7_layer_call_fn_388980

args_0
identity╓
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_3877862
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
я
h
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_388888

args_0
identity▓
max_pooling2d_1/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
ЪM
╩
F__inference_sequential_layer_call_and_return_conditional_losses_388330

inputs/
module_wrapper_388290:@#
module_wrapper_388292:@1
module_wrapper_3_388297:@@%
module_wrapper_3_388299:@2
module_wrapper_6_388304:@А&
module_wrapper_6_388306:	А+
module_wrapper_10_388312:	А$@&
module_wrapper_10_388314:@*
module_wrapper_12_388318:@ &
module_wrapper_12_388320: *
module_wrapper_14_388324: &
module_wrapper_14_388326:
identityИв&module_wrapper/StatefulPartitionedCallв)module_wrapper_10/StatefulPartitionedCallв)module_wrapper_11/StatefulPartitionedCallв)module_wrapper_12/StatefulPartitionedCallв)module_wrapper_13/StatefulPartitionedCallв)module_wrapper_14/StatefulPartitionedCallв(module_wrapper_2/StatefulPartitionedCallв(module_wrapper_3/StatefulPartitionedCallв(module_wrapper_5/StatefulPartitionedCallв(module_wrapper_6/StatefulPartitionedCallв(module_wrapper_8/StatefulPartitionedCall╜
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_388290module_wrapper_388292*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_3882492(
&module_wrapper/StatefulPartitionedCallа
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_3882232"
 module_wrapper_1/PartitionedCall▓
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_3882072*
(module_wrapper_2/StatefulPartitionedCallЄ
(module_wrapper_3/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0module_wrapper_3_388297module_wrapper_3_388299*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_3881802*
(module_wrapper_3/StatefulPartitionedCallв
 module_wrapper_4/PartitionedCallPartitionedCall1module_wrapper_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_3881542"
 module_wrapper_4/PartitionedCall▌
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0)^module_wrapper_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_3881382*
(module_wrapper_5/StatefulPartitionedCallє
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_5/StatefulPartitionedCall:output:0module_wrapper_6_388304module_wrapper_6_388306*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_3881112*
(module_wrapper_6/StatefulPartitionedCallг
 module_wrapper_7/PartitionedCallPartitionedCall1module_wrapper_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_3880852"
 module_wrapper_7/PartitionedCall▐
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0)^module_wrapper_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_3880692*
(module_wrapper_8/StatefulPartitionedCallЫ
 module_wrapper_9/PartitionedCallPartitionedCall1module_wrapper_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_3880462"
 module_wrapper_9/PartitionedCallч
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_388312module_wrapper_10_388314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_3880252+
)module_wrapper_10/StatefulPartitionedCallс
)module_wrapper_11/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_10/StatefulPartitionedCall:output:0)^module_wrapper_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_3879992+
)module_wrapper_11/StatefulPartitionedCallЁ
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_11/StatefulPartitionedCall:output:0module_wrapper_12_388318module_wrapper_12_388320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_3879722+
)module_wrapper_12/StatefulPartitionedCallт
)module_wrapper_13/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*^module_wrapper_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_3879462+
)module_wrapper_13/StatefulPartitionedCallЁ
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_13/StatefulPartitionedCall:output:0module_wrapper_14_388324module_wrapper_14_388326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_3879192+
)module_wrapper_14/StatefulPartitionedCallт
IdentityIdentity2module_wrapper_14/StatefulPartitionedCall:output:0'^module_wrapper/StatefulPartitionedCall*^module_wrapper_10/StatefulPartitionedCall*^module_wrapper_11/StatefulPartitionedCall*^module_wrapper_12/StatefulPartitionedCall*^module_wrapper_13/StatefulPartitionedCall*^module_wrapper_14/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall)^module_wrapper_3/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall)^module_wrapper_6/StatefulPartitionedCall)^module_wrapper_8/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2V
)module_wrapper_10/StatefulPartitionedCall)module_wrapper_10/StatefulPartitionedCall2V
)module_wrapper_11/StatefulPartitionedCall)module_wrapper_11/StatefulPartitionedCall2V
)module_wrapper_12/StatefulPartitionedCall)module_wrapper_12/StatefulPartitionedCall2V
)module_wrapper_13/StatefulPartitionedCall)module_wrapper_13/StatefulPartitionedCall2V
)module_wrapper_14/StatefulPartitionedCall)module_wrapper_14/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2T
(module_wrapper_3/StatefulPartitionedCall(module_wrapper_3/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall2T
(module_wrapper_6/StatefulPartitionedCall(module_wrapper_6/StatefulPartitionedCall2T
(module_wrapper_8/StatefulPartitionedCall(module_wrapper_8/StatefulPartitionedCall:W S
/
_output_shapes
:         00
 
_user_specified_nameinputs
Бa
─
F__inference_sequential_layer_call_and_return_conditional_losses_388602

inputsN
4module_wrapper_conv2d_conv2d_readvariableop_resource:@C
5module_wrapper_conv2d_biasadd_readvariableop_resource:@R
8module_wrapper_3_conv2d_1_conv2d_readvariableop_resource:@@G
9module_wrapper_3_conv2d_1_biasadd_readvariableop_resource:@S
8module_wrapper_6_conv2d_2_conv2d_readvariableop_resource:@АH
9module_wrapper_6_conv2d_2_biasadd_readvariableop_resource:	АI
6module_wrapper_10_dense_matmul_readvariableop_resource:	А$@E
7module_wrapper_10_dense_biasadd_readvariableop_resource:@J
8module_wrapper_12_dense_1_matmul_readvariableop_resource:@ G
9module_wrapper_12_dense_1_biasadd_readvariableop_resource: J
8module_wrapper_14_dense_2_matmul_readvariableop_resource: G
9module_wrapper_14_dense_2_biasadd_readvariableop_resource:
identityИв,module_wrapper/conv2d/BiasAdd/ReadVariableOpв+module_wrapper/conv2d/Conv2D/ReadVariableOpв.module_wrapper_10/dense/BiasAdd/ReadVariableOpв-module_wrapper_10/dense/MatMul/ReadVariableOpв0module_wrapper_12/dense_1/BiasAdd/ReadVariableOpв/module_wrapper_12/dense_1/MatMul/ReadVariableOpв0module_wrapper_14/dense_2/BiasAdd/ReadVariableOpв/module_wrapper_14/dense_2/MatMul/ReadVariableOpв0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpв/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOpв0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpв/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp╫
+module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp4module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+module_wrapper/conv2d/Conv2D/ReadVariableOpх
module_wrapper/conv2d/Conv2DConv2Dinputs3module_wrapper/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@*
paddingSAME*
strides
2
module_wrapper/conv2d/Conv2D╬
,module_wrapper/conv2d/BiasAdd/ReadVariableOpReadVariableOp5module_wrapper_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,module_wrapper/conv2d/BiasAdd/ReadVariableOpр
module_wrapper/conv2d/BiasAddBiasAdd%module_wrapper/conv2d/Conv2D:output:04module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@2
module_wrapper/conv2d/BiasAddв
module_wrapper/conv2d/ReluRelu&module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         00@2
module_wrapper/conv2d/ReluЄ
&module_wrapper_1/max_pooling2d/MaxPoolMaxPool(module_wrapper/conv2d/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2(
&module_wrapper_1/max_pooling2d/MaxPool╜
!module_wrapper_2/dropout/IdentityIdentity/module_wrapper_1/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_2/dropout/Identityу
/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_3_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOpХ
 module_wrapper_3/conv2d_1/Conv2DConv2D*module_wrapper_2/dropout/Identity:output:07module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2"
 module_wrapper_3/conv2d_1/Conv2D┌
0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_3_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpЁ
!module_wrapper_3/conv2d_1/BiasAddBiasAdd)module_wrapper_3/conv2d_1/Conv2D:output:08module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_3/conv2d_1/BiasAddо
module_wrapper_3/conv2d_1/ReluRelu*module_wrapper_3/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2 
module_wrapper_3/conv2d_1/Relu·
(module_wrapper_4/max_pooling2d_1/MaxPoolMaxPool,module_wrapper_3/conv2d_1/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2*
(module_wrapper_4/max_pooling2d_1/MaxPool├
#module_wrapper_5/dropout_1/IdentityIdentity1module_wrapper_4/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2%
#module_wrapper_5/dropout_1/Identityф
/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_6_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype021
/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOpШ
 module_wrapper_6/conv2d_2/Conv2DConv2D,module_wrapper_5/dropout_1/Identity:output:07module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2"
 module_wrapper_6/conv2d_2/Conv2D█
0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_6_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpё
!module_wrapper_6/conv2d_2/BiasAddBiasAdd)module_wrapper_6/conv2d_2/Conv2D:output:08module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2#
!module_wrapper_6/conv2d_2/BiasAddп
module_wrapper_6/conv2d_2/ReluRelu*module_wrapper_6/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2 
module_wrapper_6/conv2d_2/Relu√
(module_wrapper_7/max_pooling2d_2/MaxPoolMaxPool,module_wrapper_6/conv2d_2/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2*
(module_wrapper_7/max_pooling2d_2/MaxPool─
#module_wrapper_8/dropout_2/IdentityIdentity1module_wrapper_7/max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:         А2%
#module_wrapper_8/dropout_2/IdentityС
module_wrapper_9/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
module_wrapper_9/flatten/Const┘
 module_wrapper_9/flatten/ReshapeReshape,module_wrapper_8/dropout_2/Identity:output:0'module_wrapper_9/flatten/Const:output:0*
T0*(
_output_shapes
:         А$2"
 module_wrapper_9/flatten/Reshape╓
-module_wrapper_10/dense/MatMul/ReadVariableOpReadVariableOp6module_wrapper_10_dense_matmul_readvariableop_resource*
_output_shapes
:	А$@*
dtype02/
-module_wrapper_10/dense/MatMul/ReadVariableOp▐
module_wrapper_10/dense/MatMulMatMul)module_wrapper_9/flatten/Reshape:output:05module_wrapper_10/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2 
module_wrapper_10/dense/MatMul╘
.module_wrapper_10/dense/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_10_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.module_wrapper_10/dense/BiasAdd/ReadVariableOpс
module_wrapper_10/dense/BiasAddBiasAdd(module_wrapper_10/dense/MatMul:product:06module_wrapper_10/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2!
module_wrapper_10/dense/BiasAddа
module_wrapper_10/dense/ReluRelu(module_wrapper_10/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
module_wrapper_10/dense/Relu╢
$module_wrapper_11/dropout_3/IdentityIdentity*module_wrapper_10/dense/Relu:activations:0*
T0*'
_output_shapes
:         @2&
$module_wrapper_11/dropout_3/Identity█
/module_wrapper_12/dense_1/MatMul/ReadVariableOpReadVariableOp8module_wrapper_12_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype021
/module_wrapper_12/dense_1/MatMul/ReadVariableOpш
 module_wrapper_12/dense_1/MatMulMatMul-module_wrapper_11/dropout_3/Identity:output:07module_wrapper_12/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2"
 module_wrapper_12/dense_1/MatMul┌
0module_wrapper_12/dense_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_12_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0module_wrapper_12/dense_1/BiasAdd/ReadVariableOpщ
!module_wrapper_12/dense_1/BiasAddBiasAdd*module_wrapper_12/dense_1/MatMul:product:08module_wrapper_12/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2#
!module_wrapper_12/dense_1/BiasAddж
module_wrapper_12/dense_1/ReluRelu*module_wrapper_12/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2 
module_wrapper_12/dense_1/Relu╕
$module_wrapper_13/dropout_4/IdentityIdentity,module_wrapper_12/dense_1/Relu:activations:0*
T0*'
_output_shapes
:          2&
$module_wrapper_13/dropout_4/Identity█
/module_wrapper_14/dense_2/MatMul/ReadVariableOpReadVariableOp8module_wrapper_14_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype021
/module_wrapper_14/dense_2/MatMul/ReadVariableOpш
 module_wrapper_14/dense_2/MatMulMatMul-module_wrapper_13/dropout_4/Identity:output:07module_wrapper_14/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 module_wrapper_14/dense_2/MatMul┌
0module_wrapper_14/dense_2/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_14_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0module_wrapper_14/dense_2/BiasAdd/ReadVariableOpщ
!module_wrapper_14/dense_2/BiasAddBiasAdd*module_wrapper_14/dense_2/MatMul:product:08module_wrapper_14/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!module_wrapper_14/dense_2/BiasAddп
!module_wrapper_14/dense_2/SoftmaxSoftmax*module_wrapper_14/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2#
!module_wrapper_14/dense_2/Softmax╤
IdentityIdentity+module_wrapper_14/dense_2/Softmax:softmax:0-^module_wrapper/conv2d/BiasAdd/ReadVariableOp,^module_wrapper/conv2d/Conv2D/ReadVariableOp/^module_wrapper_10/dense/BiasAdd/ReadVariableOp.^module_wrapper_10/dense/MatMul/ReadVariableOp1^module_wrapper_12/dense_1/BiasAdd/ReadVariableOp0^module_wrapper_12/dense_1/MatMul/ReadVariableOp1^module_wrapper_14/dense_2/BiasAdd/ReadVariableOp0^module_wrapper_14/dense_2/MatMul/ReadVariableOp1^module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp0^module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp1^module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp0^module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 2\
,module_wrapper/conv2d/BiasAdd/ReadVariableOp,module_wrapper/conv2d/BiasAdd/ReadVariableOp2Z
+module_wrapper/conv2d/Conv2D/ReadVariableOp+module_wrapper/conv2d/Conv2D/ReadVariableOp2`
.module_wrapper_10/dense/BiasAdd/ReadVariableOp.module_wrapper_10/dense/BiasAdd/ReadVariableOp2^
-module_wrapper_10/dense/MatMul/ReadVariableOp-module_wrapper_10/dense/MatMul/ReadVariableOp2d
0module_wrapper_12/dense_1/BiasAdd/ReadVariableOp0module_wrapper_12/dense_1/BiasAdd/ReadVariableOp2b
/module_wrapper_12/dense_1/MatMul/ReadVariableOp/module_wrapper_12/dense_1/MatMul/ReadVariableOp2d
0module_wrapper_14/dense_2/BiasAdd/ReadVariableOp0module_wrapper_14/dense_2/BiasAdd/ReadVariableOp2b
/module_wrapper_14/dense_2/MatMul/ReadVariableOp/module_wrapper_14/dense_2/MatMul/ReadVariableOp2d
0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp2b
/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp2d
0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp2b
/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         00
 
_user_specified_nameinputs
щ
h
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_388796

args_0
identityо
max_pooling2d/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolz
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         00@:W S
/
_output_shapes
:         00@
 
_user_specified_nameargs_0
Ч
л
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_388947

args_0B
'conv2d_2_conv2d_readvariableop_resource:@А7
(conv2d_2_biasadd_readvariableop_resource:	А
identityИвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOp▒
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_2/Conv2D/ReadVariableOp┐
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_2/Conv2Dи
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpн
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_2/Relu╗
IdentityIdentityconv2d_2/Relu:activations:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
б
h
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_389024

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstА
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         А$2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
О
i
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_387825

args_0
identityn
dropout_3/IdentityIdentityargs_0*
T0*'
_output_shapes
:         @2
dropout_3/Identityo
IdentityIdentitydropout_3/Identity:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
О
i
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_389146

args_0
identityn
dropout_4/IdentityIdentityargs_0*
T0*'
_output_shapes
:          2
dropout_4/Identityo
IdentityIdentitydropout_4/Identity:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
╖
l
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_389158

args_0
identityИw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_4/dropout/ConstС
dropout_4/dropout/MulMulargs_0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout_4/dropout/Mulh
dropout_4/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_4/dropout/Shapeт
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seedО╠ЮЇ20
.dropout_4/dropout/random_uniform/RandomUniformЙ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2"
 dropout_4/dropout/GreaterEqual/yц
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2 
dropout_4/dropout/GreaterEqualЭ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_4/dropout/Castв
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_4/dropout/Mul_1o
IdentityIdentitydropout_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
я
M
1__inference_module_wrapper_5_layer_call_fn_388920

args_0
identity╒
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_3877622
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
є
M
1__inference_module_wrapper_8_layer_call_fn_389007

args_0
identity╓
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_3877932
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
 
j
1__inference_module_wrapper_8_layer_call_fn_389012

args_0
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_3880692
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
■
k
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_389002

args_0
identityИw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/ConstЪ
dropout_2/dropout/MulMulargs_0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:         А2
dropout_2/dropout/Mulh
dropout_2/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeы
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype0*
seedО╠ЮЇ20
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yя
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А2 
dropout_2/dropout/GreaterEqualж
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2
dropout_2/dropout/Castл
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2
dropout_2/dropout/Mul_1x
IdentityIdentitydropout_2/dropout/Mul_1:z:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╓
J
.__inference_max_pooling2d_layer_call_fn_388522

inputs
identityэ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3885162
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
О
i
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_387849

args_0
identityn
dropout_4/IdentityIdentityargs_0*
T0*'
_output_shapes
:          2
dropout_4/Identityo
IdentityIdentitydropout_4/Identity:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
│
а
2__inference_module_wrapper_10_layer_call_fn_389074

args_0
unknown:	А$@
	unknown_0:@
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_3880252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А$
 
_user_specified_nameargs_0
н
h
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_387762

args_0
identityv
dropout_1/IdentityIdentityargs_0*
T0*/
_output_shapes
:         @2
dropout_1/Identityw
IdentityIdentitydropout_1/Identity:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Юа
─
F__inference_sequential_layer_call_and_return_conditional_losses_388693

inputsN
4module_wrapper_conv2d_conv2d_readvariableop_resource:@C
5module_wrapper_conv2d_biasadd_readvariableop_resource:@R
8module_wrapper_3_conv2d_1_conv2d_readvariableop_resource:@@G
9module_wrapper_3_conv2d_1_biasadd_readvariableop_resource:@S
8module_wrapper_6_conv2d_2_conv2d_readvariableop_resource:@АH
9module_wrapper_6_conv2d_2_biasadd_readvariableop_resource:	АI
6module_wrapper_10_dense_matmul_readvariableop_resource:	А$@E
7module_wrapper_10_dense_biasadd_readvariableop_resource:@J
8module_wrapper_12_dense_1_matmul_readvariableop_resource:@ G
9module_wrapper_12_dense_1_biasadd_readvariableop_resource: J
8module_wrapper_14_dense_2_matmul_readvariableop_resource: G
9module_wrapper_14_dense_2_biasadd_readvariableop_resource:
identityИв,module_wrapper/conv2d/BiasAdd/ReadVariableOpв+module_wrapper/conv2d/Conv2D/ReadVariableOpв.module_wrapper_10/dense/BiasAdd/ReadVariableOpв-module_wrapper_10/dense/MatMul/ReadVariableOpв0module_wrapper_12/dense_1/BiasAdd/ReadVariableOpв/module_wrapper_12/dense_1/MatMul/ReadVariableOpв0module_wrapper_14/dense_2/BiasAdd/ReadVariableOpв/module_wrapper_14/dense_2/MatMul/ReadVariableOpв0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpв/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOpв0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpв/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp╫
+module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp4module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+module_wrapper/conv2d/Conv2D/ReadVariableOpх
module_wrapper/conv2d/Conv2DConv2Dinputs3module_wrapper/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@*
paddingSAME*
strides
2
module_wrapper/conv2d/Conv2D╬
,module_wrapper/conv2d/BiasAdd/ReadVariableOpReadVariableOp5module_wrapper_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,module_wrapper/conv2d/BiasAdd/ReadVariableOpр
module_wrapper/conv2d/BiasAddBiasAdd%module_wrapper/conv2d/Conv2D:output:04module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@2
module_wrapper/conv2d/BiasAddв
module_wrapper/conv2d/ReluRelu&module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         00@2
module_wrapper/conv2d/ReluЄ
&module_wrapper_1/max_pooling2d/MaxPoolMaxPool(module_wrapper/conv2d/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2(
&module_wrapper_1/max_pooling2d/MaxPoolХ
&module_wrapper_2/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&module_wrapper_2/dropout/dropout/Constя
$module_wrapper_2/dropout/dropout/MulMul/module_wrapper_1/max_pooling2d/MaxPool:output:0/module_wrapper_2/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         @2&
$module_wrapper_2/dropout/dropout/Mulп
&module_wrapper_2/dropout/dropout/ShapeShape/module_wrapper_1/max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2(
&module_wrapper_2/dropout/dropout/ShapeЧ
=module_wrapper_2/dropout/dropout/random_uniform/RandomUniformRandomUniform/module_wrapper_2/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ2?
=module_wrapper_2/dropout/dropout/random_uniform/RandomUniformз
/module_wrapper_2/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/module_wrapper_2/dropout/dropout/GreaterEqual/yк
-module_wrapper_2/dropout/dropout/GreaterEqualGreaterEqualFmodule_wrapper_2/dropout/dropout/random_uniform/RandomUniform:output:08module_wrapper_2/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2/
-module_wrapper_2/dropout/dropout/GreaterEqual╥
%module_wrapper_2/dropout/dropout/CastCast1module_wrapper_2/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2'
%module_wrapper_2/dropout/dropout/Castц
&module_wrapper_2/dropout/dropout/Mul_1Mul(module_wrapper_2/dropout/dropout/Mul:z:0)module_wrapper_2/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2(
&module_wrapper_2/dropout/dropout/Mul_1у
/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_3_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype021
/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOpХ
 module_wrapper_3/conv2d_1/Conv2DConv2D*module_wrapper_2/dropout/dropout/Mul_1:z:07module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2"
 module_wrapper_3/conv2d_1/Conv2D┌
0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_3_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype022
0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpЁ
!module_wrapper_3/conv2d_1/BiasAddBiasAdd)module_wrapper_3/conv2d_1/Conv2D:output:08module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2#
!module_wrapper_3/conv2d_1/BiasAddо
module_wrapper_3/conv2d_1/ReluRelu*module_wrapper_3/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2 
module_wrapper_3/conv2d_1/Relu·
(module_wrapper_4/max_pooling2d_1/MaxPoolMaxPool,module_wrapper_3/conv2d_1/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2*
(module_wrapper_4/max_pooling2d_1/MaxPoolЩ
(module_wrapper_5/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2*
(module_wrapper_5/dropout_1/dropout/Constў
&module_wrapper_5/dropout_1/dropout/MulMul1module_wrapper_4/max_pooling2d_1/MaxPool:output:01module_wrapper_5/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:         @2(
&module_wrapper_5/dropout_1/dropout/Mul╡
(module_wrapper_5/dropout_1/dropout/ShapeShape1module_wrapper_4/max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2*
(module_wrapper_5/dropout_1/dropout/Shapeк
?module_wrapper_5/dropout_1/dropout/random_uniform/RandomUniformRandomUniform1module_wrapper_5/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ*
seed22A
?module_wrapper_5/dropout_1/dropout/random_uniform/RandomUniformл
1module_wrapper_5/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?23
1module_wrapper_5/dropout_1/dropout/GreaterEqual/y▓
/module_wrapper_5/dropout_1/dropout/GreaterEqualGreaterEqualHmodule_wrapper_5/dropout_1/dropout/random_uniform/RandomUniform:output:0:module_wrapper_5/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @21
/module_wrapper_5/dropout_1/dropout/GreaterEqual╪
'module_wrapper_5/dropout_1/dropout/CastCast3module_wrapper_5/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2)
'module_wrapper_5/dropout_1/dropout/Castю
(module_wrapper_5/dropout_1/dropout/Mul_1Mul*module_wrapper_5/dropout_1/dropout/Mul:z:0+module_wrapper_5/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2*
(module_wrapper_5/dropout_1/dropout/Mul_1ф
/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOpReadVariableOp8module_wrapper_6_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype021
/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOpШ
 module_wrapper_6/conv2d_2/Conv2DConv2D,module_wrapper_5/dropout_1/dropout/Mul_1:z:07module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2"
 module_wrapper_6/conv2d_2/Conv2D█
0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_6_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype022
0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpё
!module_wrapper_6/conv2d_2/BiasAddBiasAdd)module_wrapper_6/conv2d_2/Conv2D:output:08module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2#
!module_wrapper_6/conv2d_2/BiasAddп
module_wrapper_6/conv2d_2/ReluRelu*module_wrapper_6/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2 
module_wrapper_6/conv2d_2/Relu√
(module_wrapper_7/max_pooling2d_2/MaxPoolMaxPool,module_wrapper_6/conv2d_2/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
2*
(module_wrapper_7/max_pooling2d_2/MaxPoolЩ
(module_wrapper_8/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2*
(module_wrapper_8/dropout_2/dropout/Const°
&module_wrapper_8/dropout_2/dropout/MulMul1module_wrapper_7/max_pooling2d_2/MaxPool:output:01module_wrapper_8/dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:         А2(
&module_wrapper_8/dropout_2/dropout/Mul╡
(module_wrapper_8/dropout_2/dropout/ShapeShape1module_wrapper_7/max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
:2*
(module_wrapper_8/dropout_2/dropout/Shapeл
?module_wrapper_8/dropout_2/dropout/random_uniform/RandomUniformRandomUniform1module_wrapper_8/dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:         А*
dtype0*
seedО╠ЮЇ*
seed22A
?module_wrapper_8/dropout_2/dropout/random_uniform/RandomUniformл
1module_wrapper_8/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?23
1module_wrapper_8/dropout_2/dropout/GreaterEqual/y│
/module_wrapper_8/dropout_2/dropout/GreaterEqualGreaterEqualHmodule_wrapper_8/dropout_2/dropout/random_uniform/RandomUniform:output:0:module_wrapper_8/dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:         А21
/module_wrapper_8/dropout_2/dropout/GreaterEqual┘
'module_wrapper_8/dropout_2/dropout/CastCast3module_wrapper_8/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:         А2)
'module_wrapper_8/dropout_2/dropout/Castя
(module_wrapper_8/dropout_2/dropout/Mul_1Mul*module_wrapper_8/dropout_2/dropout/Mul:z:0+module_wrapper_8/dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:         А2*
(module_wrapper_8/dropout_2/dropout/Mul_1С
module_wrapper_9/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
module_wrapper_9/flatten/Const┘
 module_wrapper_9/flatten/ReshapeReshape,module_wrapper_8/dropout_2/dropout/Mul_1:z:0'module_wrapper_9/flatten/Const:output:0*
T0*(
_output_shapes
:         А$2"
 module_wrapper_9/flatten/Reshape╓
-module_wrapper_10/dense/MatMul/ReadVariableOpReadVariableOp6module_wrapper_10_dense_matmul_readvariableop_resource*
_output_shapes
:	А$@*
dtype02/
-module_wrapper_10/dense/MatMul/ReadVariableOp▐
module_wrapper_10/dense/MatMulMatMul)module_wrapper_9/flatten/Reshape:output:05module_wrapper_10/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2 
module_wrapper_10/dense/MatMul╘
.module_wrapper_10/dense/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_10_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.module_wrapper_10/dense/BiasAdd/ReadVariableOpс
module_wrapper_10/dense/BiasAddBiasAdd(module_wrapper_10/dense/MatMul:product:06module_wrapper_10/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2!
module_wrapper_10/dense/BiasAddа
module_wrapper_10/dense/ReluRelu(module_wrapper_10/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2
module_wrapper_10/dense/ReluЫ
)module_wrapper_11/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2+
)module_wrapper_11/dropout_3/dropout/Constы
'module_wrapper_11/dropout_3/dropout/MulMul*module_wrapper_10/dense/Relu:activations:02module_wrapper_11/dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:         @2)
'module_wrapper_11/dropout_3/dropout/Mul░
)module_wrapper_11/dropout_3/dropout/ShapeShape*module_wrapper_10/dense/Relu:activations:0*
T0*
_output_shapes
:2+
)module_wrapper_11/dropout_3/dropout/Shapeе
@module_wrapper_11/dropout_3/dropout/random_uniform/RandomUniformRandomUniform2module_wrapper_11/dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ*
seed22B
@module_wrapper_11/dropout_3/dropout/random_uniform/RandomUniformн
2module_wrapper_11/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>24
2module_wrapper_11/dropout_3/dropout/GreaterEqual/yо
0module_wrapper_11/dropout_3/dropout/GreaterEqualGreaterEqualImodule_wrapper_11/dropout_3/dropout/random_uniform/RandomUniform:output:0;module_wrapper_11/dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @22
0module_wrapper_11/dropout_3/dropout/GreaterEqual╙
(module_wrapper_11/dropout_3/dropout/CastCast4module_wrapper_11/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2*
(module_wrapper_11/dropout_3/dropout/Castъ
)module_wrapper_11/dropout_3/dropout/Mul_1Mul+module_wrapper_11/dropout_3/dropout/Mul:z:0,module_wrapper_11/dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2+
)module_wrapper_11/dropout_3/dropout/Mul_1█
/module_wrapper_12/dense_1/MatMul/ReadVariableOpReadVariableOp8module_wrapper_12_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype021
/module_wrapper_12/dense_1/MatMul/ReadVariableOpш
 module_wrapper_12/dense_1/MatMulMatMul-module_wrapper_11/dropout_3/dropout/Mul_1:z:07module_wrapper_12/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2"
 module_wrapper_12/dense_1/MatMul┌
0module_wrapper_12/dense_1/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_12_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0module_wrapper_12/dense_1/BiasAdd/ReadVariableOpщ
!module_wrapper_12/dense_1/BiasAddBiasAdd*module_wrapper_12/dense_1/MatMul:product:08module_wrapper_12/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2#
!module_wrapper_12/dense_1/BiasAddж
module_wrapper_12/dense_1/ReluRelu*module_wrapper_12/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2 
module_wrapper_12/dense_1/ReluЫ
)module_wrapper_13/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2+
)module_wrapper_13/dropout_4/dropout/Constэ
'module_wrapper_13/dropout_4/dropout/MulMul,module_wrapper_12/dense_1/Relu:activations:02module_wrapper_13/dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:          2)
'module_wrapper_13/dropout_4/dropout/Mul▓
)module_wrapper_13/dropout_4/dropout/ShapeShape,module_wrapper_12/dense_1/Relu:activations:0*
T0*
_output_shapes
:2+
)module_wrapper_13/dropout_4/dropout/Shapeе
@module_wrapper_13/dropout_4/dropout/random_uniform/RandomUniformRandomUniform2module_wrapper_13/dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0*
seedО╠ЮЇ*
seed22B
@module_wrapper_13/dropout_4/dropout/random_uniform/RandomUniformн
2module_wrapper_13/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>24
2module_wrapper_13/dropout_4/dropout/GreaterEqual/yо
0module_wrapper_13/dropout_4/dropout/GreaterEqualGreaterEqualImodule_wrapper_13/dropout_4/dropout/random_uniform/RandomUniform:output:0;module_wrapper_13/dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          22
0module_wrapper_13/dropout_4/dropout/GreaterEqual╙
(module_wrapper_13/dropout_4/dropout/CastCast4module_wrapper_13/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2*
(module_wrapper_13/dropout_4/dropout/Castъ
)module_wrapper_13/dropout_4/dropout/Mul_1Mul+module_wrapper_13/dropout_4/dropout/Mul:z:0,module_wrapper_13/dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:          2+
)module_wrapper_13/dropout_4/dropout/Mul_1█
/module_wrapper_14/dense_2/MatMul/ReadVariableOpReadVariableOp8module_wrapper_14_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype021
/module_wrapper_14/dense_2/MatMul/ReadVariableOpш
 module_wrapper_14/dense_2/MatMulMatMul-module_wrapper_13/dropout_4/dropout/Mul_1:z:07module_wrapper_14/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 module_wrapper_14/dense_2/MatMul┌
0module_wrapper_14/dense_2/BiasAdd/ReadVariableOpReadVariableOp9module_wrapper_14_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0module_wrapper_14/dense_2/BiasAdd/ReadVariableOpщ
!module_wrapper_14/dense_2/BiasAddBiasAdd*module_wrapper_14/dense_2/MatMul:product:08module_wrapper_14/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!module_wrapper_14/dense_2/BiasAddп
!module_wrapper_14/dense_2/SoftmaxSoftmax*module_wrapper_14/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2#
!module_wrapper_14/dense_2/Softmax╤
IdentityIdentity+module_wrapper_14/dense_2/Softmax:softmax:0-^module_wrapper/conv2d/BiasAdd/ReadVariableOp,^module_wrapper/conv2d/Conv2D/ReadVariableOp/^module_wrapper_10/dense/BiasAdd/ReadVariableOp.^module_wrapper_10/dense/MatMul/ReadVariableOp1^module_wrapper_12/dense_1/BiasAdd/ReadVariableOp0^module_wrapper_12/dense_1/MatMul/ReadVariableOp1^module_wrapper_14/dense_2/BiasAdd/ReadVariableOp0^module_wrapper_14/dense_2/MatMul/ReadVariableOp1^module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp0^module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp1^module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp0^module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 2\
,module_wrapper/conv2d/BiasAdd/ReadVariableOp,module_wrapper/conv2d/BiasAdd/ReadVariableOp2Z
+module_wrapper/conv2d/Conv2D/ReadVariableOp+module_wrapper/conv2d/Conv2D/ReadVariableOp2`
.module_wrapper_10/dense/BiasAdd/ReadVariableOp.module_wrapper_10/dense/BiasAdd/ReadVariableOp2^
-module_wrapper_10/dense/MatMul/ReadVariableOp-module_wrapper_10/dense/MatMul/ReadVariableOp2d
0module_wrapper_12/dense_1/BiasAdd/ReadVariableOp0module_wrapper_12/dense_1/BiasAdd/ReadVariableOp2b
/module_wrapper_12/dense_1/MatMul/ReadVariableOp/module_wrapper_12/dense_1/MatMul/ReadVariableOp2d
0module_wrapper_14/dense_2/BiasAdd/ReadVariableOp0module_wrapper_14/dense_2/BiasAdd/ReadVariableOp2b
/module_wrapper_14/dense_2/MatMul/ReadVariableOp/module_wrapper_14/dense_2/MatMul/ReadVariableOp2d
0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp0module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp2b
/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp2d
0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp0module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp2b
/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         00
 
_user_specified_nameinputs
▄
Ч
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_389056

args_07
$dense_matmul_readvariableop_resource:	А$@3
%dense_biasadd_readvariableop_resource:@
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А$@*
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2

dense/Reluй
IdentityIdentitydense/Relu:activations:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А$: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А$
 
_user_specified_nameargs_0
Ў
k
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_388915

args_0
identityИw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/ConstЩ
dropout_1/dropout/MulMulargs_0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout_1/dropout/Mulh
dropout_1/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeъ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ20
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yю
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2 
dropout_1/dropout/GreaterEqualе
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout_1/dropout/Castк
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout_1/dropout/Mul_1w
IdentityIdentitydropout_1/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
√
j
1__inference_module_wrapper_5_layer_call_fn_388925

args_0
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_3881382
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ч
л
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_388936

args_0B
'conv2d_2_conv2d_readvariableop_resource:@А7
(conv2d_2_biasadd_readvariableop_resource:	А
identityИвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOp▒
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_2/Conv2D/ReadVariableOp┐
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_2/Conv2Dи
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpн
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_2/Relu╗
IdentityIdentityconv2d_2/Relu:activations:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
░
Я
2__inference_module_wrapper_12_layer_call_fn_389141

args_0
unknown:@ 
	unknown_0: 
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_3879722
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
є

┬
+__inference_sequential_layer_call_fn_388722

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А$@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3878692
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         00
 
_user_specified_nameinputs
П
й
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_387744

args_0A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@
identityИвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╛
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Relu║
IdentityIdentityconv2d_1/Relu:activations:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
▌
k
2__inference_module_wrapper_13_layer_call_fn_389168

args_0
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_3879462
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
П
й
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_388860

args_0A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@
identityИвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOp░
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp╛
conv2d_1/Conv2DConv2Dargs_0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2
conv2d_1/Conv2Dз
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpм
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
conv2d_1/Relu║
IdentityIdentityconv2d_1/Relu:activations:0 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
▄
Ч
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_387814

args_07
$dense_matmul_readvariableop_resource:	А$@3
%dense_biasadd_readvariableop_resource:@
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	А$@*
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2

dense/Reluй
IdentityIdentitydense/Relu:activations:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А$: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:         А$
 
_user_specified_nameargs_0
╫
Я
J__inference_module_wrapper_layer_call_and_return_conditional_losses_388762

args_0?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╕
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@*
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         00@2
conv2d/Relu┤
IdentityIdentityconv2d/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         00
 
_user_specified_nameargs_0
▒
h
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_388990

args_0
identityw
dropout_2/IdentityIdentityargs_0*
T0*0
_output_shapes
:         А2
dropout_2/Identityx
IdentityIdentitydropout_2/Identity:output:0*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
╞
k
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_388207

args_0
identityИs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/ConstУ
dropout/dropout/MulMulargs_0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/dropout/Muld
dropout/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout/dropout/Shapeф
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ2.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yц
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/dropout/GreaterEqualЯ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/dropout/Castв
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/dropout/Mul_1u
IdentityIdentitydropout/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
я
M
1__inference_module_wrapper_1_layer_call_fn_388806

args_0
identity╒
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_3877242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         00@:W S
/
_output_shapes
:         00@
 
_user_specified_nameargs_0
╫
Я
J__inference_module_wrapper_layer_call_and_return_conditional_losses_388773

args_0?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpк
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp╕
conv2d/Conv2DConv2Dargs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@*
paddingSAME*
strides
2
conv2d/Conv2Dб
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOpд
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         00@2
conv2d/Relu┤
IdentityIdentityconv2d/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00: : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         00
 
_user_specified_nameargs_0
╩╔
┼ 
"__inference__traced_restore_389511
file_prefix&
assignvariableop_adamax_iter:	 *
 assignvariableop_1_adamax_beta_1: *
 assignvariableop_2_adamax_beta_2: )
assignvariableop_3_adamax_decay: 1
'assignvariableop_4_adamax_learning_rate: I
/assignvariableop_5_module_wrapper_conv2d_kernel:@;
-assignvariableop_6_module_wrapper_conv2d_bias:@M
3assignvariableop_7_module_wrapper_3_conv2d_1_kernel:@@?
1assignvariableop_8_module_wrapper_3_conv2d_1_bias:@N
3assignvariableop_9_module_wrapper_6_conv2d_2_kernel:@АA
2assignvariableop_10_module_wrapper_6_conv2d_2_bias:	АE
2assignvariableop_11_module_wrapper_10_dense_kernel:	А$@>
0assignvariableop_12_module_wrapper_10_dense_bias:@F
4assignvariableop_13_module_wrapper_12_dense_1_kernel:@ @
2assignvariableop_14_module_wrapper_12_dense_1_bias: F
4assignvariableop_15_module_wrapper_14_dense_2_kernel: @
2assignvariableop_16_module_wrapper_14_dense_2_bias:#
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: S
9assignvariableop_21_adamax_module_wrapper_conv2d_kernel_m:@E
7assignvariableop_22_adamax_module_wrapper_conv2d_bias_m:@W
=assignvariableop_23_adamax_module_wrapper_3_conv2d_1_kernel_m:@@I
;assignvariableop_24_adamax_module_wrapper_3_conv2d_1_bias_m:@X
=assignvariableop_25_adamax_module_wrapper_6_conv2d_2_kernel_m:@АJ
;assignvariableop_26_adamax_module_wrapper_6_conv2d_2_bias_m:	АN
;assignvariableop_27_adamax_module_wrapper_10_dense_kernel_m:	А$@G
9assignvariableop_28_adamax_module_wrapper_10_dense_bias_m:@O
=assignvariableop_29_adamax_module_wrapper_12_dense_1_kernel_m:@ I
;assignvariableop_30_adamax_module_wrapper_12_dense_1_bias_m: O
=assignvariableop_31_adamax_module_wrapper_14_dense_2_kernel_m: I
;assignvariableop_32_adamax_module_wrapper_14_dense_2_bias_m:S
9assignvariableop_33_adamax_module_wrapper_conv2d_kernel_v:@E
7assignvariableop_34_adamax_module_wrapper_conv2d_bias_v:@W
=assignvariableop_35_adamax_module_wrapper_3_conv2d_1_kernel_v:@@I
;assignvariableop_36_adamax_module_wrapper_3_conv2d_1_bias_v:@X
=assignvariableop_37_adamax_module_wrapper_6_conv2d_2_kernel_v:@АJ
;assignvariableop_38_adamax_module_wrapper_6_conv2d_2_bias_v:	АN
;assignvariableop_39_adamax_module_wrapper_10_dense_kernel_v:	А$@G
9assignvariableop_40_adamax_module_wrapper_10_dense_bias_v:@O
=assignvariableop_41_adamax_module_wrapper_12_dense_1_kernel_v:@ I
;assignvariableop_42_adamax_module_wrapper_12_dense_1_bias_v: O
=assignvariableop_43_adamax_module_wrapper_14_dense_2_kernel_v: I
;assignvariableop_44_adamax_module_wrapper_14_dense_2_bias_v:
identity_46ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Т
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ю
valueФBС.B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesФ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╬
_output_shapes╗
╕::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

IdentityЫ
AssignVariableOpAssignVariableOpassignvariableop_adamax_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_adamax_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2е
AssignVariableOp_2AssignVariableOp assignvariableop_2_adamax_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOpassignvariableop_3_adamax_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4м
AssignVariableOp_4AssignVariableOp'assignvariableop_4_adamax_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5┤
AssignVariableOp_5AssignVariableOp/assignvariableop_5_module_wrapper_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6▓
AssignVariableOp_6AssignVariableOp-assignvariableop_6_module_wrapper_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╕
AssignVariableOp_7AssignVariableOp3assignvariableop_7_module_wrapper_3_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╢
AssignVariableOp_8AssignVariableOp1assignvariableop_8_module_wrapper_3_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╕
AssignVariableOp_9AssignVariableOp3assignvariableop_9_module_wrapper_6_conv2d_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10║
AssignVariableOp_10AssignVariableOp2assignvariableop_10_module_wrapper_6_conv2d_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11║
AssignVariableOp_11AssignVariableOp2assignvariableop_11_module_wrapper_10_dense_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╕
AssignVariableOp_12AssignVariableOp0assignvariableop_12_module_wrapper_10_dense_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╝
AssignVariableOp_13AssignVariableOp4assignvariableop_13_module_wrapper_12_dense_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14║
AssignVariableOp_14AssignVariableOp2assignvariableop_14_module_wrapper_12_dense_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╝
AssignVariableOp_15AssignVariableOp4assignvariableop_15_module_wrapper_14_dense_2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16║
AssignVariableOp_16AssignVariableOp2assignvariableop_16_module_wrapper_14_dense_2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17б
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18б
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19г
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20г
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21┴
AssignVariableOp_21AssignVariableOp9assignvariableop_21_adamax_module_wrapper_conv2d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22┐
AssignVariableOp_22AssignVariableOp7assignvariableop_22_adamax_module_wrapper_conv2d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23┼
AssignVariableOp_23AssignVariableOp=assignvariableop_23_adamax_module_wrapper_3_conv2d_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24├
AssignVariableOp_24AssignVariableOp;assignvariableop_24_adamax_module_wrapper_3_conv2d_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25┼
AssignVariableOp_25AssignVariableOp=assignvariableop_25_adamax_module_wrapper_6_conv2d_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26├
AssignVariableOp_26AssignVariableOp;assignvariableop_26_adamax_module_wrapper_6_conv2d_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27├
AssignVariableOp_27AssignVariableOp;assignvariableop_27_adamax_module_wrapper_10_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28┴
AssignVariableOp_28AssignVariableOp9assignvariableop_28_adamax_module_wrapper_10_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29┼
AssignVariableOp_29AssignVariableOp=assignvariableop_29_adamax_module_wrapper_12_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30├
AssignVariableOp_30AssignVariableOp;assignvariableop_30_adamax_module_wrapper_12_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31┼
AssignVariableOp_31AssignVariableOp=assignvariableop_31_adamax_module_wrapper_14_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32├
AssignVariableOp_32AssignVariableOp;assignvariableop_32_adamax_module_wrapper_14_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33┴
AssignVariableOp_33AssignVariableOp9assignvariableop_33_adamax_module_wrapper_conv2d_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34┐
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adamax_module_wrapper_conv2d_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┼
AssignVariableOp_35AssignVariableOp=assignvariableop_35_adamax_module_wrapper_3_conv2d_1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36├
AssignVariableOp_36AssignVariableOp;assignvariableop_36_adamax_module_wrapper_3_conv2d_1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37┼
AssignVariableOp_37AssignVariableOp=assignvariableop_37_adamax_module_wrapper_6_conv2d_2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38├
AssignVariableOp_38AssignVariableOp;assignvariableop_38_adamax_module_wrapper_6_conv2d_2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39├
AssignVariableOp_39AssignVariableOp;assignvariableop_39_adamax_module_wrapper_10_dense_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40┴
AssignVariableOp_40AssignVariableOp9assignvariableop_40_adamax_module_wrapper_10_dense_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41┼
AssignVariableOp_41AssignVariableOp=assignvariableop_41_adamax_module_wrapper_12_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42├
AssignVariableOp_42AssignVariableOp;assignvariableop_42_adamax_module_wrapper_12_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43┼
AssignVariableOp_43AssignVariableOp=assignvariableop_43_adamax_module_wrapper_14_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44├
AssignVariableOp_44AssignVariableOp;assignvariableop_44_adamax_module_wrapper_14_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╝
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45п
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╞q
╡
!__inference__wrapped_model_387695
module_wrapper_inputY
?sequential_module_wrapper_conv2d_conv2d_readvariableop_resource:@N
@sequential_module_wrapper_conv2d_biasadd_readvariableop_resource:@]
Csequential_module_wrapper_3_conv2d_1_conv2d_readvariableop_resource:@@R
Dsequential_module_wrapper_3_conv2d_1_biasadd_readvariableop_resource:@^
Csequential_module_wrapper_6_conv2d_2_conv2d_readvariableop_resource:@АS
Dsequential_module_wrapper_6_conv2d_2_biasadd_readvariableop_resource:	АT
Asequential_module_wrapper_10_dense_matmul_readvariableop_resource:	А$@P
Bsequential_module_wrapper_10_dense_biasadd_readvariableop_resource:@U
Csequential_module_wrapper_12_dense_1_matmul_readvariableop_resource:@ R
Dsequential_module_wrapper_12_dense_1_biasadd_readvariableop_resource: U
Csequential_module_wrapper_14_dense_2_matmul_readvariableop_resource: R
Dsequential_module_wrapper_14_dense_2_biasadd_readvariableop_resource:
identityИв7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpв6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpв9sequential/module_wrapper_10/dense/BiasAdd/ReadVariableOpв8sequential/module_wrapper_10/dense/MatMul/ReadVariableOpв;sequential/module_wrapper_12/dense_1/BiasAdd/ReadVariableOpв:sequential/module_wrapper_12/dense_1/MatMul/ReadVariableOpв;sequential/module_wrapper_14/dense_2/BiasAdd/ReadVariableOpв:sequential/module_wrapper_14/dense_2/MatMul/ReadVariableOpв;sequential/module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpв:sequential/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOpв;sequential/module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpв:sequential/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp°
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpReadVariableOp?sequential_module_wrapper_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype028
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOpФ
'sequential/module_wrapper/conv2d/Conv2DConv2Dmodule_wrapper_input>sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@*
paddingSAME*
strides
2)
'sequential/module_wrapper/conv2d/Conv2Dя
7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpReadVariableOp@sequential_module_wrapper_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOpМ
(sequential/module_wrapper/conv2d/BiasAddBiasAdd0sequential/module_wrapper/conv2d/Conv2D:output:0?sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         00@2*
(sequential/module_wrapper/conv2d/BiasAdd├
%sequential/module_wrapper/conv2d/ReluRelu1sequential/module_wrapper/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         00@2'
%sequential/module_wrapper/conv2d/ReluУ
1sequential/module_wrapper_1/max_pooling2d/MaxPoolMaxPool3sequential/module_wrapper/conv2d/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
23
1sequential/module_wrapper_1/max_pooling2d/MaxPool▐
,sequential/module_wrapper_2/dropout/IdentityIdentity:sequential/module_wrapper_1/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         @2.
,sequential/module_wrapper_2/dropout/IdentityД
:sequential/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOpReadVariableOpCsequential_module_wrapper_3_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:sequential/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp┴
+sequential/module_wrapper_3/conv2d_1/Conv2DConv2D5sequential/module_wrapper_2/dropout/Identity:output:0Bsequential/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
2-
+sequential/module_wrapper_3/conv2d_1/Conv2D√
;sequential/module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_3_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;sequential/module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOpЬ
,sequential/module_wrapper_3/conv2d_1/BiasAddBiasAdd4sequential/module_wrapper_3/conv2d_1/Conv2D:output:0Csequential/module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2.
,sequential/module_wrapper_3/conv2d_1/BiasAdd╧
)sequential/module_wrapper_3/conv2d_1/ReluRelu5sequential/module_wrapper_3/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         @2+
)sequential/module_wrapper_3/conv2d_1/ReluЫ
3sequential/module_wrapper_4/max_pooling2d_1/MaxPoolMaxPool7sequential/module_wrapper_3/conv2d_1/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
25
3sequential/module_wrapper_4/max_pooling2d_1/MaxPoolф
.sequential/module_wrapper_5/dropout_1/IdentityIdentity<sequential/module_wrapper_4/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @20
.sequential/module_wrapper_5/dropout_1/IdentityЕ
:sequential/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOpReadVariableOpCsequential_module_wrapper_6_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02<
:sequential/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp─
+sequential/module_wrapper_6/conv2d_2/Conv2DConv2D7sequential/module_wrapper_5/dropout_1/Identity:output:0Bsequential/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2-
+sequential/module_wrapper_6/conv2d_2/Conv2D№
;sequential/module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_6_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02=
;sequential/module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOpЭ
,sequential/module_wrapper_6/conv2d_2/BiasAddBiasAdd4sequential/module_wrapper_6/conv2d_2/Conv2D:output:0Csequential/module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2.
,sequential/module_wrapper_6/conv2d_2/BiasAdd╨
)sequential/module_wrapper_6/conv2d_2/ReluRelu5sequential/module_wrapper_6/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2+
)sequential/module_wrapper_6/conv2d_2/ReluЬ
3sequential/module_wrapper_7/max_pooling2d_2/MaxPoolMaxPool7sequential/module_wrapper_6/conv2d_2/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
25
3sequential/module_wrapper_7/max_pooling2d_2/MaxPoolх
.sequential/module_wrapper_8/dropout_2/IdentityIdentity<sequential/module_wrapper_7/max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:         А20
.sequential/module_wrapper_8/dropout_2/Identityз
)sequential/module_wrapper_9/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)sequential/module_wrapper_9/flatten/ConstЕ
+sequential/module_wrapper_9/flatten/ReshapeReshape7sequential/module_wrapper_8/dropout_2/Identity:output:02sequential/module_wrapper_9/flatten/Const:output:0*
T0*(
_output_shapes
:         А$2-
+sequential/module_wrapper_9/flatten/Reshapeў
8sequential/module_wrapper_10/dense/MatMul/ReadVariableOpReadVariableOpAsequential_module_wrapper_10_dense_matmul_readvariableop_resource*
_output_shapes
:	А$@*
dtype02:
8sequential/module_wrapper_10/dense/MatMul/ReadVariableOpК
)sequential/module_wrapper_10/dense/MatMulMatMul4sequential/module_wrapper_9/flatten/Reshape:output:0@sequential/module_wrapper_10/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2+
)sequential/module_wrapper_10/dense/MatMulї
9sequential/module_wrapper_10/dense/BiasAdd/ReadVariableOpReadVariableOpBsequential_module_wrapper_10_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9sequential/module_wrapper_10/dense/BiasAdd/ReadVariableOpН
*sequential/module_wrapper_10/dense/BiasAddBiasAdd3sequential/module_wrapper_10/dense/MatMul:product:0Asequential/module_wrapper_10/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2,
*sequential/module_wrapper_10/dense/BiasAdd┴
'sequential/module_wrapper_10/dense/ReluRelu3sequential/module_wrapper_10/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         @2)
'sequential/module_wrapper_10/dense/Relu╫
/sequential/module_wrapper_11/dropout_3/IdentityIdentity5sequential/module_wrapper_10/dense/Relu:activations:0*
T0*'
_output_shapes
:         @21
/sequential/module_wrapper_11/dropout_3/Identity№
:sequential/module_wrapper_12/dense_1/MatMul/ReadVariableOpReadVariableOpCsequential_module_wrapper_12_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02<
:sequential/module_wrapper_12/dense_1/MatMul/ReadVariableOpФ
+sequential/module_wrapper_12/dense_1/MatMulMatMul8sequential/module_wrapper_11/dropout_3/Identity:output:0Bsequential/module_wrapper_12/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2-
+sequential/module_wrapper_12/dense_1/MatMul√
;sequential/module_wrapper_12/dense_1/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_12_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;sequential/module_wrapper_12/dense_1/BiasAdd/ReadVariableOpХ
,sequential/module_wrapper_12/dense_1/BiasAddBiasAdd5sequential/module_wrapper_12/dense_1/MatMul:product:0Csequential/module_wrapper_12/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2.
,sequential/module_wrapper_12/dense_1/BiasAdd╟
)sequential/module_wrapper_12/dense_1/ReluRelu5sequential/module_wrapper_12/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2+
)sequential/module_wrapper_12/dense_1/Relu┘
/sequential/module_wrapper_13/dropout_4/IdentityIdentity7sequential/module_wrapper_12/dense_1/Relu:activations:0*
T0*'
_output_shapes
:          21
/sequential/module_wrapper_13/dropout_4/Identity№
:sequential/module_wrapper_14/dense_2/MatMul/ReadVariableOpReadVariableOpCsequential_module_wrapper_14_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02<
:sequential/module_wrapper_14/dense_2/MatMul/ReadVariableOpФ
+sequential/module_wrapper_14/dense_2/MatMulMatMul8sequential/module_wrapper_13/dropout_4/Identity:output:0Bsequential/module_wrapper_14/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2-
+sequential/module_wrapper_14/dense_2/MatMul√
;sequential/module_wrapper_14/dense_2/BiasAdd/ReadVariableOpReadVariableOpDsequential_module_wrapper_14_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;sequential/module_wrapper_14/dense_2/BiasAdd/ReadVariableOpХ
,sequential/module_wrapper_14/dense_2/BiasAddBiasAdd5sequential/module_wrapper_14/dense_2/MatMul:product:0Csequential/module_wrapper_14/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2.
,sequential/module_wrapper_14/dense_2/BiasAdd╨
,sequential/module_wrapper_14/dense_2/SoftmaxSoftmax5sequential/module_wrapper_14/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2.
,sequential/module_wrapper_14/dense_2/Softmaxр
IdentityIdentity6sequential/module_wrapper_14/dense_2/Softmax:softmax:08^sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp7^sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp:^sequential/module_wrapper_10/dense/BiasAdd/ReadVariableOp9^sequential/module_wrapper_10/dense/MatMul/ReadVariableOp<^sequential/module_wrapper_12/dense_1/BiasAdd/ReadVariableOp;^sequential/module_wrapper_12/dense_1/MatMul/ReadVariableOp<^sequential/module_wrapper_14/dense_2/BiasAdd/ReadVariableOp;^sequential/module_wrapper_14/dense_2/MatMul/ReadVariableOp<^sequential/module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp;^sequential/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp<^sequential/module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp;^sequential/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 2r
7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp7sequential/module_wrapper/conv2d/BiasAdd/ReadVariableOp2p
6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp6sequential/module_wrapper/conv2d/Conv2D/ReadVariableOp2v
9sequential/module_wrapper_10/dense/BiasAdd/ReadVariableOp9sequential/module_wrapper_10/dense/BiasAdd/ReadVariableOp2t
8sequential/module_wrapper_10/dense/MatMul/ReadVariableOp8sequential/module_wrapper_10/dense/MatMul/ReadVariableOp2z
;sequential/module_wrapper_12/dense_1/BiasAdd/ReadVariableOp;sequential/module_wrapper_12/dense_1/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_12/dense_1/MatMul/ReadVariableOp:sequential/module_wrapper_12/dense_1/MatMul/ReadVariableOp2z
;sequential/module_wrapper_14/dense_2/BiasAdd/ReadVariableOp;sequential/module_wrapper_14/dense_2/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_14/dense_2/MatMul/ReadVariableOp:sequential/module_wrapper_14/dense_2/MatMul/ReadVariableOp2z
;sequential/module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp;sequential/module_wrapper_3/conv2d_1/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp:sequential/module_wrapper_3/conv2d_1/Conv2D/ReadVariableOp2z
;sequential/module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp;sequential/module_wrapper_6/conv2d_2/BiasAdd/ReadVariableOp2x
:sequential/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp:sequential/module_wrapper_6/conv2d_2/Conv2D/ReadVariableOp:e a
/
_output_shapes
:         00
.
_user_specified_namemodule_wrapper_input
б
h
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_389018

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstА
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         А$2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
я
h
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_387755

args_0
identity▓
max_pooling2d_1/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
╤
N
2__inference_module_wrapper_11_layer_call_fn_389096

args_0
identity╬
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_3878252
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
╓
ж
1__inference_module_wrapper_3_layer_call_fn_388878

args_0!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_3881802
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
╞
k
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_388828

args_0
identityИs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/ConstУ
dropout/dropout/MulMulargs_0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/dropout/Muld
dropout/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout/dropout/Shapeф
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ2.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yц
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/dropout/GreaterEqualЯ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/dropout/Castв
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/dropout/Mul_1u
IdentityIdentitydropout/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
щ
h
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_388801

args_0
identityо
max_pooling2d/MaxPoolMaxPoolargs_0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolz
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         00@:W S
/
_output_shapes
:         00@
 
_user_specified_nameargs_0
Ч
л
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_388111

args_0B
'conv2d_2_conv2d_readvariableop_resource:@А7
(conv2d_2_biasadd_readvariableop_resource:	А
identityИвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOp▒
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_2/Conv2D/ReadVariableOp┐
conv2d_2/Conv2DConv2Dargs_0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
2
conv2d_2/Conv2Dи
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpн
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
conv2d_2/Relu╗
IdentityIdentityconv2d_2/Relu:activations:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
б
h
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_388046

args_0
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten/ConstА
flatten/ReshapeReshapeargs_0flatten/Const:output:0*
T0*(
_output_shapes
:         А$2
flatten/Reshapem
IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:         А$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameargs_0
О
Ю
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_387838

args_08
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOpЛ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/Reluп
IdentityIdentitydense_1/Relu:activations:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
я
M
1__inference_module_wrapper_4_layer_call_fn_388893

args_0
identity╒
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_3877552
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
Ц
Ю
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_389190

args_08
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:
identityИвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpе
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOpЛ
dense_2/MatMulMatMulargs_0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_2/Softmaxо
IdentityIdentitydense_2/Softmax:softmax:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameargs_0
─M
╪
F__inference_sequential_layer_call_and_return_conditional_losses_388472
module_wrapper_input/
module_wrapper_388432:@#
module_wrapper_388434:@1
module_wrapper_3_388439:@@%
module_wrapper_3_388441:@2
module_wrapper_6_388446:@А&
module_wrapper_6_388448:	А+
module_wrapper_10_388454:	А$@&
module_wrapper_10_388456:@*
module_wrapper_12_388460:@ &
module_wrapper_12_388462: *
module_wrapper_14_388466: &
module_wrapper_14_388468:
identityИв&module_wrapper/StatefulPartitionedCallв)module_wrapper_10/StatefulPartitionedCallв)module_wrapper_11/StatefulPartitionedCallв)module_wrapper_12/StatefulPartitionedCallв)module_wrapper_13/StatefulPartitionedCallв)module_wrapper_14/StatefulPartitionedCallв(module_wrapper_2/StatefulPartitionedCallв(module_wrapper_3/StatefulPartitionedCallв(module_wrapper_5/StatefulPartitionedCallв(module_wrapper_6/StatefulPartitionedCallв(module_wrapper_8/StatefulPartitionedCall╦
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_388432module_wrapper_388434*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_3882492(
&module_wrapper/StatefulPartitionedCallа
 module_wrapper_1/PartitionedCallPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_3882232"
 module_wrapper_1/PartitionedCall▓
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_3882072*
(module_wrapper_2/StatefulPartitionedCallЄ
(module_wrapper_3/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0module_wrapper_3_388439module_wrapper_3_388441*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_3881802*
(module_wrapper_3/StatefulPartitionedCallв
 module_wrapper_4/PartitionedCallPartitionedCall1module_wrapper_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_3881542"
 module_wrapper_4/PartitionedCall▌
(module_wrapper_5/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_4/PartitionedCall:output:0)^module_wrapper_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_3881382*
(module_wrapper_5/StatefulPartitionedCallє
(module_wrapper_6/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_5/StatefulPartitionedCall:output:0module_wrapper_6_388446module_wrapper_6_388448*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_3881112*
(module_wrapper_6/StatefulPartitionedCallг
 module_wrapper_7/PartitionedCallPartitionedCall1module_wrapper_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_3880852"
 module_wrapper_7/PartitionedCall▐
(module_wrapper_8/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_7/PartitionedCall:output:0)^module_wrapper_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_3880692*
(module_wrapper_8/StatefulPartitionedCallЫ
 module_wrapper_9/PartitionedCallPartitionedCall1module_wrapper_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_3880462"
 module_wrapper_9/PartitionedCallч
)module_wrapper_10/StatefulPartitionedCallStatefulPartitionedCall)module_wrapper_9/PartitionedCall:output:0module_wrapper_10_388454module_wrapper_10_388456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_3880252+
)module_wrapper_10/StatefulPartitionedCallс
)module_wrapper_11/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_10/StatefulPartitionedCall:output:0)^module_wrapper_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_3879992+
)module_wrapper_11/StatefulPartitionedCallЁ
)module_wrapper_12/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_11/StatefulPartitionedCall:output:0module_wrapper_12_388460module_wrapper_12_388462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_3879722+
)module_wrapper_12/StatefulPartitionedCallт
)module_wrapper_13/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_12/StatefulPartitionedCall:output:0*^module_wrapper_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_3879462+
)module_wrapper_13/StatefulPartitionedCallЁ
)module_wrapper_14/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_13/StatefulPartitionedCall:output:0module_wrapper_14_388466module_wrapper_14_388468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_3879192+
)module_wrapper_14/StatefulPartitionedCallт
IdentityIdentity2module_wrapper_14/StatefulPartitionedCall:output:0'^module_wrapper/StatefulPartitionedCall*^module_wrapper_10/StatefulPartitionedCall*^module_wrapper_11/StatefulPartitionedCall*^module_wrapper_12/StatefulPartitionedCall*^module_wrapper_13/StatefulPartitionedCall*^module_wrapper_14/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall)^module_wrapper_3/StatefulPartitionedCall)^module_wrapper_5/StatefulPartitionedCall)^module_wrapper_6/StatefulPartitionedCall)^module_wrapper_8/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2V
)module_wrapper_10/StatefulPartitionedCall)module_wrapper_10/StatefulPartitionedCall2V
)module_wrapper_11/StatefulPartitionedCall)module_wrapper_11/StatefulPartitionedCall2V
)module_wrapper_12/StatefulPartitionedCall)module_wrapper_12/StatefulPartitionedCall2V
)module_wrapper_13/StatefulPartitionedCall)module_wrapper_13/StatefulPartitionedCall2V
)module_wrapper_14/StatefulPartitionedCall)module_wrapper_14/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2T
(module_wrapper_3/StatefulPartitionedCall(module_wrapper_3/StatefulPartitionedCall2T
(module_wrapper_5/StatefulPartitionedCall(module_wrapper_5/StatefulPartitionedCall2T
(module_wrapper_6/StatefulPartitionedCall(module_wrapper_6/StatefulPartitionedCall2T
(module_wrapper_8/StatefulPartitionedCall(module_wrapper_8/StatefulPartitionedCall:e a
/
_output_shapes
:         00
.
_user_specified_namemodule_wrapper_input
Ў
k
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_388138

args_0
identityИw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/ConstЩ
dropout_1/dropout/MulMulargs_0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout_1/dropout/Mulh
dropout_1/dropout/ShapeShapeargs_0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeъ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype0*
seedО╠ЮЇ20
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yю
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2 
dropout_1/dropout/GreaterEqualе
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout_1/dropout/Castк
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout_1/dropout/Mul_1w
IdentityIdentitydropout_1/dropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
л
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_388528

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
є

┬
+__inference_sequential_layer_call_fn_388751

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А
	unknown_5:	А$@
	unknown_6:@
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10:
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3883302
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:         00: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         00
 
_user_specified_nameinputs
О
Ю
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_389123

args_08
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_1/MatMul/ReadVariableOpЛ
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/Reluп
IdentityIdentitydense_1/Relu:activations:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameargs_0
╥
д
/__inference_module_wrapper_layer_call_fn_388791

args_0!
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_3882492
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         00@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         00
 
_user_specified_nameargs_0
╓
ж
1__inference_module_wrapper_3_layer_call_fn_388869

args_0!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_3877442
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0
┌
и
1__inference_module_wrapper_6_layer_call_fn_388956

args_0"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_3877752
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameargs_0"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╓
serving_default┬
]
module_wrapper_inputE
&serving_default_module_wrapper_input:0         00E
module_wrapper_140
StatefulPartitionedCall:0         tensorflow/serving/predict:Я╧
ъ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
layer-13
layer_with_weights-5
layer-14
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+ь&call_and_return_all_conditional_losses
э_default_save_signature
ю__call__"╕
_tf_keras_sequentialЩ{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "module_wrapper_input"}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}, {"class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}]}, "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 48, 48, 1]}, "float32", "module_wrapper_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 2}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adamax", "config": {"name": "Adamax", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.9800000190734863, "beta_2": 0.9800000190734863, "epsilon": 1e-07}}}}
╗
_module
	variables
trainable_variables
regularization_losses
	keras_api
+я&call_and_return_all_conditional_losses
Ё__call__"Э
_tf_keras_layerГ{"name": "module_wrapper", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
_module
	variables
trainable_variables
regularization_losses
	keras_api
+ё&call_and_return_all_conditional_losses
Є__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
 _module
!	variables
"trainable_variables
#regularization_losses
$	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
%_module
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+ї&call_and_return_all_conditional_losses
Ў__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
*_module
+	variables
,trainable_variables
-regularization_losses
.	keras_api
+ў&call_and_return_all_conditional_losses
°__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
/_module
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+∙&call_and_return_all_conditional_losses
·__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
4_module
5	variables
6trainable_variables
7regularization_losses
8	keras_api
+√&call_and_return_all_conditional_losses
№__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
9_module
:	variables
;trainable_variables
<regularization_losses
=	keras_api
+¤&call_and_return_all_conditional_losses
■__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
>_module
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
+ &call_and_return_all_conditional_losses
А__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╜
C_module
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"Я
_tf_keras_layerЕ{"name": "module_wrapper_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
H_module
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"а
_tf_keras_layerЖ{"name": "module_wrapper_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
M_module
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"а
_tf_keras_layerЖ{"name": "module_wrapper_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
R_module
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
+З&call_and_return_all_conditional_losses
И__call__"а
_tf_keras_layerЖ{"name": "module_wrapper_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
W_module
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"а
_tf_keras_layerЖ{"name": "module_wrapper_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
╛
\_module
]	variables
^trainable_variables
_regularization_losses
`	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"а
_tf_keras_layerЖ{"name": "module_wrapper_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ModuleWrapper", "config": {"layer was saved without config": true}}
├
aiter

bbeta_1

cbeta_2
	ddecay
elearning_ratefm╘gm╒hm╓im╫jm╪km┘lm┌mm█nm▄om▌pm▐qm▀fvрgvсhvтivуjvфkvхlvцmvчnvшovщpvъqvы"
	optimizer
v
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11"
trackable_list_wrapper
╬
rnon_trainable_variables

slayers
trainable_variables
regularization_losses
tlayer_metrics
umetrics
	variables
vlayer_regularization_losses
ю__call__
э_default_save_signature
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
-
Нserving_default"
signature_map
я


fkernel
gbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
+О&call_and_return_all_conditional_losses
П__call__"╚	
_tf_keras_layerо	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 1]}}
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables
{non_trainable_variables
|layer_metrics
trainable_variables
regularization_losses
}metrics

~layers
layer_regularization_losses
Ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
Б
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"ь
_tf_keras_layer╥{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
	variables
Дnon_trainable_variables
Еlayer_metrics
trainable_variables
regularization_losses
Жmetrics
Зlayers
 Иlayer_regularization_losses
Є__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
ч
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"╥
_tf_keras_layer╕{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
!	variables
Нnon_trainable_variables
Оlayer_metrics
"trainable_variables
#regularization_losses
Пmetrics
Рlayers
 Сlayer_regularization_losses
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
°	

hkernel
ibias
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"═
_tf_keras_layer│{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 64]}}
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
&	variables
Цnon_trainable_variables
Чlayer_metrics
'trainable_variables
(regularization_losses
Шmetrics
Щlayers
 Ъlayer_regularization_losses
Ў__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
Е
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"Ё
_tf_keras_layer╓{"name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
+	variables
Яnon_trainable_variables
аlayer_metrics
,trainable_variables
-regularization_losses
бmetrics
вlayers
 гlayer_regularization_losses
°__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
ы
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"╓
_tf_keras_layer╝{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
0	variables
иnon_trainable_variables
йlayer_metrics
1trainable_variables
2regularization_losses
кmetrics
лlayers
 мlayer_regularization_losses
·__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
_generic_user_object
∙	

jkernel
kbias
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"╬
_tf_keras_layer┤{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 64]}}
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
5	variables
▒non_trainable_variables
▓layer_metrics
6trainable_variables
7regularization_losses
│metrics
┤layers
 ╡layer_regularization_losses
№__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
Е
╢	variables
╖trainable_variables
╕regularization_losses
╣	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"Ё
_tf_keras_layer╓{"name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
:	variables
║non_trainable_variables
╗layer_metrics
;trainable_variables
<regularization_losses
╝metrics
╜layers
 ╛layer_regularization_losses
■__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
ы
┐	variables
└trainable_variables
┴regularization_losses
┬	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"╓
_tf_keras_layer╝{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
?	variables
├non_trainable_variables
─layer_metrics
@trainable_variables
Aregularization_losses
┼metrics
╞layers
 ╟layer_regularization_losses
А__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
ш
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
+а&call_and_return_all_conditional_losses
б__call__"╙
_tf_keras_layer╣{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
D	variables
╠non_trainable_variables
═layer_metrics
Etrainable_variables
Fregularization_losses
╬metrics
╧layers
 ╨layer_regularization_losses
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
Ў

lkernel
mbias
╤	variables
╥trainable_variables
╙regularization_losses
╘	keras_api
+в&call_and_return_all_conditional_losses
г__call__"╦
_tf_keras_layer▒{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4608}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4608]}}
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
I	variables
╒non_trainable_variables
╓layer_metrics
Jtrainable_variables
Kregularization_losses
╫metrics
╪layers
 ┘layer_regularization_losses
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
ь
┌	variables
█trainable_variables
▄regularization_losses
▌	keras_api
+д&call_and_return_all_conditional_losses
е__call__"╫
_tf_keras_layer╜{"name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
N	variables
▐non_trainable_variables
▀layer_metrics
Otrainable_variables
Pregularization_losses
рmetrics
сlayers
 тlayer_regularization_losses
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
Ў

nkernel
obias
у	variables
фtrainable_variables
хregularization_losses
ц	keras_api
+ж&call_and_return_all_conditional_losses
з__call__"╦
_tf_keras_layer▒{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
S	variables
чnon_trainable_variables
шlayer_metrics
Ttrainable_variables
Uregularization_losses
щmetrics
ъlayers
 ыlayer_regularization_losses
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
ь
ь	variables
эtrainable_variables
юregularization_losses
я	keras_api
+и&call_and_return_all_conditional_losses
й__call__"╫
_tf_keras_layer╜{"name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
X	variables
Ёnon_trainable_variables
ёlayer_metrics
Ytrainable_variables
Zregularization_losses
Єmetrics
єlayers
 Їlayer_regularization_losses
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
°

pkernel
qbias
ї	variables
Ўtrainable_variables
ўregularization_losses
°	keras_api
+к&call_and_return_all_conditional_losses
л__call__"═
_tf_keras_layer│{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
]	variables
∙non_trainable_variables
·layer_metrics
^trainable_variables
_regularization_losses
√metrics
№layers
 ¤layer_regularization_losses
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adamax/iter
: (2Adamax/beta_1
: (2Adamax/beta_2
: (2Adamax/decay
: (2Adamax/learning_rate
6:4@2module_wrapper/conv2d/kernel
(:&@2module_wrapper/conv2d/bias
::8@@2 module_wrapper_3/conv2d_1/kernel
,:*@2module_wrapper_3/conv2d_1/bias
;:9@А2 module_wrapper_6/conv2d_2/kernel
-:+А2module_wrapper_6/conv2d_2/bias
1:/	А$@2module_wrapper_10/dense/kernel
*:(@2module_wrapper_10/dense/bias
2:0@ 2 module_wrapper_12/dense_1/kernel
,:* 2module_wrapper_12/dense_1/bias
2:0 2 module_wrapper_14/dense_2/kernel
,:*2module_wrapper_14/dense_2/bias
 "
trackable_list_wrapper
О
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
■0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
w	variables
Аnon_trainable_variables
Бlayer_metrics
xtrainable_variables
yregularization_losses
Вmetrics
Гlayers
 Дlayer_regularization_losses
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
А	variables
Еnon_trainable_variables
Жlayer_metrics
Бtrainable_variables
Вregularization_losses
Зmetrics
Иlayers
 Йlayer_regularization_losses
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Й	variables
Кnon_trainable_variables
Лlayer_metrics
Кtrainable_variables
Лregularization_losses
Мmetrics
Нlayers
 Оlayer_regularization_losses
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Т	variables
Пnon_trainable_variables
Рlayer_metrics
Уtrainable_variables
Фregularization_losses
Сmetrics
Тlayers
 Уlayer_regularization_losses
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ы	variables
Фnon_trainable_variables
Хlayer_metrics
Ьtrainable_variables
Эregularization_losses
Цmetrics
Чlayers
 Шlayer_regularization_losses
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
д	variables
Щnon_trainable_variables
Ъlayer_metrics
еtrainable_variables
жregularization_losses
Ыmetrics
Ьlayers
 Эlayer_regularization_losses
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
н	variables
Юnon_trainable_variables
Яlayer_metrics
оtrainable_variables
пregularization_losses
аmetrics
бlayers
 вlayer_regularization_losses
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╢	variables
гnon_trainable_variables
дlayer_metrics
╖trainable_variables
╕regularization_losses
еmetrics
жlayers
 зlayer_regularization_losses
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┐	variables
иnon_trainable_variables
йlayer_metrics
└trainable_variables
┴regularization_losses
кmetrics
лlayers
 мlayer_regularization_losses
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╚	variables
нnon_trainable_variables
оlayer_metrics
╔trainable_variables
╩regularization_losses
пmetrics
░layers
 ▒layer_regularization_losses
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╤	variables
▓non_trainable_variables
│layer_metrics
╥trainable_variables
╙regularization_losses
┤metrics
╡layers
 ╢layer_regularization_losses
г__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┌	variables
╖non_trainable_variables
╕layer_metrics
█trainable_variables
▄regularization_losses
╣metrics
║layers
 ╗layer_regularization_losses
е__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
у	variables
╝non_trainable_variables
╜layer_metrics
фtrainable_variables
хregularization_losses
╛metrics
┐layers
 └layer_regularization_losses
з__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ь	variables
┴non_trainable_variables
┬layer_metrics
эtrainable_variables
юregularization_losses
├metrics
─layers
 ┼layer_regularization_losses
й__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ї	variables
╞non_trainable_variables
╟layer_metrics
Ўtrainable_variables
ўregularization_losses
╚metrics
╔layers
 ╩layer_regularization_losses
л__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╫

╦total

╠count
═	variables
╬	keras_api"Ь
_tf_keras_metricБ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 3}
Ы

╧total

╨count
╤
_fn_kwargs
╥	variables
╙	keras_api"╧
_tf_keras_metric┤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 2}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
╦0
╠1"
trackable_list_wrapper
.
═	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╧0
╨1"
trackable_list_wrapper
.
╥	variables"
_generic_user_object
=:;@2%Adamax/module_wrapper/conv2d/kernel/m
/:-@2#Adamax/module_wrapper/conv2d/bias/m
A:?@@2)Adamax/module_wrapper_3/conv2d_1/kernel/m
3:1@2'Adamax/module_wrapper_3/conv2d_1/bias/m
B:@@А2)Adamax/module_wrapper_6/conv2d_2/kernel/m
4:2А2'Adamax/module_wrapper_6/conv2d_2/bias/m
8:6	А$@2'Adamax/module_wrapper_10/dense/kernel/m
1:/@2%Adamax/module_wrapper_10/dense/bias/m
9:7@ 2)Adamax/module_wrapper_12/dense_1/kernel/m
3:1 2'Adamax/module_wrapper_12/dense_1/bias/m
9:7 2)Adamax/module_wrapper_14/dense_2/kernel/m
3:12'Adamax/module_wrapper_14/dense_2/bias/m
=:;@2%Adamax/module_wrapper/conv2d/kernel/v
/:-@2#Adamax/module_wrapper/conv2d/bias/v
A:?@@2)Adamax/module_wrapper_3/conv2d_1/kernel/v
3:1@2'Adamax/module_wrapper_3/conv2d_1/bias/v
B:@@А2)Adamax/module_wrapper_6/conv2d_2/kernel/v
4:2А2'Adamax/module_wrapper_6/conv2d_2/bias/v
8:6	А$@2'Adamax/module_wrapper_10/dense/kernel/v
1:/@2%Adamax/module_wrapper_10/dense/bias/v
9:7@ 2)Adamax/module_wrapper_12/dense_1/kernel/v
3:1 2'Adamax/module_wrapper_12/dense_1/bias/v
9:7 2)Adamax/module_wrapper_14/dense_2/kernel/v
3:12'Adamax/module_wrapper_14/dense_2/bias/v
ц2у
F__inference_sequential_layer_call_and_return_conditional_losses_388602
F__inference_sequential_layer_call_and_return_conditional_losses_388693
F__inference_sequential_layer_call_and_return_conditional_losses_388429
F__inference_sequential_layer_call_and_return_conditional_losses_388472└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ї2ё
!__inference__wrapped_model_387695╦
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *;в8
6К3
module_wrapper_input         00
·2ў
+__inference_sequential_layer_call_fn_387896
+__inference_sequential_layer_call_fn_388722
+__inference_sequential_layer_call_fn_388751
+__inference_sequential_layer_call_fn_388386└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
J__inference_module_wrapper_layer_call_and_return_conditional_losses_388762
J__inference_module_wrapper_layer_call_and_return_conditional_losses_388773└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
и2е
/__inference_module_wrapper_layer_call_fn_388782
/__inference_module_wrapper_layer_call_fn_388791└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_388796
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_388801└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_1_layer_call_fn_388806
1__inference_module_wrapper_1_layer_call_fn_388811└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_388816
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_388828└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_2_layer_call_fn_388833
1__inference_module_wrapper_2_layer_call_fn_388838└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_388849
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_388860└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_3_layer_call_fn_388869
1__inference_module_wrapper_3_layer_call_fn_388878└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_388883
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_388888└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_4_layer_call_fn_388893
1__inference_module_wrapper_4_layer_call_fn_388898└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_388903
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_388915└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_5_layer_call_fn_388920
1__inference_module_wrapper_5_layer_call_fn_388925└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_388936
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_388947└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_6_layer_call_fn_388956
1__inference_module_wrapper_6_layer_call_fn_388965└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_388970
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_388975└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_7_layer_call_fn_388980
1__inference_module_wrapper_7_layer_call_fn_388985└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_388990
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_389002└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_8_layer_call_fn_389007
1__inference_module_wrapper_8_layer_call_fn_389012└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
т2▀
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_389018
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_389024└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
м2й
1__inference_module_wrapper_9_layer_call_fn_389029
1__inference_module_wrapper_9_layer_call_fn_389034└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ф2с
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_389045
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_389056└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
о2л
2__inference_module_wrapper_10_layer_call_fn_389065
2__inference_module_wrapper_10_layer_call_fn_389074└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ф2с
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_389079
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_389091└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
о2л
2__inference_module_wrapper_11_layer_call_fn_389096
2__inference_module_wrapper_11_layer_call_fn_389101└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ф2с
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_389112
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_389123└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
о2л
2__inference_module_wrapper_12_layer_call_fn_389132
2__inference_module_wrapper_12_layer_call_fn_389141└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ф2с
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_389146
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_389158└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
о2л
2__inference_module_wrapper_13_layer_call_fn_389163
2__inference_module_wrapper_13_layer_call_fn_389168└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ф2с
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_389179
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_389190└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
о2л
2__inference_module_wrapper_14_layer_call_fn_389199
2__inference_module_wrapper_14_layer_call_fn_389208└
╖▓│
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╪B╒
$__inference_signature_wrapper_388509module_wrapper_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▒2о
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_388516р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ц2У
.__inference_max_pooling2d_layer_call_fn_388522р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
│2░
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_388528р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_1_layer_call_fn_388534р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
│2░
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_388540р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_2_layer_call_fn_388546р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
║2╖┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ┬
!__inference__wrapped_model_387695ЬfghijklmnopqEвB
;в8
6К3
module_wrapper_input         00
к "EкB
@
module_wrapper_14+К(
module_wrapper_14         ю
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_388528ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_1_layer_call_fn_388534СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_388540ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_2_layer_call_fn_388546СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ь
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_388516ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ─
.__inference_max_pooling2d_layer_call_fn_388522СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╛
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_389045mlm@в=
&в#
!К
args_0         А$
к

trainingp "%в"
К
0         @
Ъ ╛
M__inference_module_wrapper_10_layer_call_and_return_conditional_losses_389056mlm@в=
&в#
!К
args_0         А$
к

trainingp"%в"
К
0         @
Ъ Ц
2__inference_module_wrapper_10_layer_call_fn_389065`lm@в=
&в#
!К
args_0         А$
к

trainingp "К         @Ц
2__inference_module_wrapper_10_layer_call_fn_389074`lm@в=
&в#
!К
args_0         А$
к

trainingp"К         @╣
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_389079h?в<
%в"
 К
args_0         @
к

trainingp "%в"
К
0         @
Ъ ╣
M__inference_module_wrapper_11_layer_call_and_return_conditional_losses_389091h?в<
%в"
 К
args_0         @
к

trainingp"%в"
К
0         @
Ъ С
2__inference_module_wrapper_11_layer_call_fn_389096[?в<
%в"
 К
args_0         @
к

trainingp "К         @С
2__inference_module_wrapper_11_layer_call_fn_389101[?в<
%в"
 К
args_0         @
к

trainingp"К         @╜
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_389112lno?в<
%в"
 К
args_0         @
к

trainingp "%в"
К
0          
Ъ ╜
M__inference_module_wrapper_12_layer_call_and_return_conditional_losses_389123lno?в<
%в"
 К
args_0         @
к

trainingp"%в"
К
0          
Ъ Х
2__inference_module_wrapper_12_layer_call_fn_389132_no?в<
%в"
 К
args_0         @
к

trainingp "К          Х
2__inference_module_wrapper_12_layer_call_fn_389141_no?в<
%в"
 К
args_0         @
к

trainingp"К          ╣
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_389146h?в<
%в"
 К
args_0          
к

trainingp "%в"
К
0          
Ъ ╣
M__inference_module_wrapper_13_layer_call_and_return_conditional_losses_389158h?в<
%в"
 К
args_0          
к

trainingp"%в"
К
0          
Ъ С
2__inference_module_wrapper_13_layer_call_fn_389163[?в<
%в"
 К
args_0          
к

trainingp "К          С
2__inference_module_wrapper_13_layer_call_fn_389168[?в<
%в"
 К
args_0          
к

trainingp"К          ╜
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_389179lpq?в<
%в"
 К
args_0          
к

trainingp "%в"
К
0         
Ъ ╜
M__inference_module_wrapper_14_layer_call_and_return_conditional_losses_389190lpq?в<
%в"
 К
args_0          
к

trainingp"%в"
К
0         
Ъ Х
2__inference_module_wrapper_14_layer_call_fn_389199_pq?в<
%в"
 К
args_0          
к

trainingp "К         Х
2__inference_module_wrapper_14_layer_call_fn_389208_pq?в<
%в"
 К
args_0          
к

trainingp"К         ╚
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_388796xGвD
-в*
(К%
args_0         00@
к

trainingp "-в*
#К 
0         @
Ъ ╚
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_388801xGвD
-в*
(К%
args_0         00@
к

trainingp"-в*
#К 
0         @
Ъ а
1__inference_module_wrapper_1_layer_call_fn_388806kGвD
-в*
(К%
args_0         00@
к

trainingp " К         @а
1__inference_module_wrapper_1_layer_call_fn_388811kGвD
-в*
(К%
args_0         00@
к

trainingp" К         @╚
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_388816xGвD
-в*
(К%
args_0         @
к

trainingp "-в*
#К 
0         @
Ъ ╚
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_388828xGвD
-в*
(К%
args_0         @
к

trainingp"-в*
#К 
0         @
Ъ а
1__inference_module_wrapper_2_layer_call_fn_388833kGвD
-в*
(К%
args_0         @
к

trainingp " К         @а
1__inference_module_wrapper_2_layer_call_fn_388838kGвD
-в*
(К%
args_0         @
к

trainingp" К         @╠
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_388849|hiGвD
-в*
(К%
args_0         @
к

trainingp "-в*
#К 
0         @
Ъ ╠
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_388860|hiGвD
-в*
(К%
args_0         @
к

trainingp"-в*
#К 
0         @
Ъ д
1__inference_module_wrapper_3_layer_call_fn_388869ohiGвD
-в*
(К%
args_0         @
к

trainingp " К         @д
1__inference_module_wrapper_3_layer_call_fn_388878ohiGвD
-в*
(К%
args_0         @
к

trainingp" К         @╚
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_388883xGвD
-в*
(К%
args_0         @
к

trainingp "-в*
#К 
0         @
Ъ ╚
L__inference_module_wrapper_4_layer_call_and_return_conditional_losses_388888xGвD
-в*
(К%
args_0         @
к

trainingp"-в*
#К 
0         @
Ъ а
1__inference_module_wrapper_4_layer_call_fn_388893kGвD
-в*
(К%
args_0         @
к

trainingp " К         @а
1__inference_module_wrapper_4_layer_call_fn_388898kGвD
-в*
(К%
args_0         @
к

trainingp" К         @╚
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_388903xGвD
-в*
(К%
args_0         @
к

trainingp "-в*
#К 
0         @
Ъ ╚
L__inference_module_wrapper_5_layer_call_and_return_conditional_losses_388915xGвD
-в*
(К%
args_0         @
к

trainingp"-в*
#К 
0         @
Ъ а
1__inference_module_wrapper_5_layer_call_fn_388920kGвD
-в*
(К%
args_0         @
к

trainingp " К         @а
1__inference_module_wrapper_5_layer_call_fn_388925kGвD
-в*
(К%
args_0         @
к

trainingp" К         @═
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_388936}jkGвD
-в*
(К%
args_0         @
к

trainingp ".в+
$К!
0         А
Ъ ═
L__inference_module_wrapper_6_layer_call_and_return_conditional_losses_388947}jkGвD
-в*
(К%
args_0         @
к

trainingp".в+
$К!
0         А
Ъ е
1__inference_module_wrapper_6_layer_call_fn_388956pjkGвD
-в*
(К%
args_0         @
к

trainingp "!К         Ае
1__inference_module_wrapper_6_layer_call_fn_388965pjkGвD
-в*
(К%
args_0         @
к

trainingp"!К         А╩
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_388970zHвE
.в+
)К&
args_0         А
к

trainingp ".в+
$К!
0         А
Ъ ╩
L__inference_module_wrapper_7_layer_call_and_return_conditional_losses_388975zHвE
.в+
)К&
args_0         А
к

trainingp".в+
$К!
0         А
Ъ в
1__inference_module_wrapper_7_layer_call_fn_388980mHвE
.в+
)К&
args_0         А
к

trainingp "!К         Ав
1__inference_module_wrapper_7_layer_call_fn_388985mHвE
.в+
)К&
args_0         А
к

trainingp"!К         А╩
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_388990zHвE
.в+
)К&
args_0         А
к

trainingp ".в+
$К!
0         А
Ъ ╩
L__inference_module_wrapper_8_layer_call_and_return_conditional_losses_389002zHвE
.в+
)К&
args_0         А
к

trainingp".в+
$К!
0         А
Ъ в
1__inference_module_wrapper_8_layer_call_fn_389007mHвE
.в+
)К&
args_0         А
к

trainingp "!К         Ав
1__inference_module_wrapper_8_layer_call_fn_389012mHвE
.в+
)К&
args_0         А
к

trainingp"!К         А┬
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_389018rHвE
.в+
)К&
args_0         А
к

trainingp "&в#
К
0         А$
Ъ ┬
L__inference_module_wrapper_9_layer_call_and_return_conditional_losses_389024rHвE
.в+
)К&
args_0         А
к

trainingp"&в#
К
0         А$
Ъ Ъ
1__inference_module_wrapper_9_layer_call_fn_389029eHвE
.в+
)К&
args_0         А
к

trainingp "К         А$Ъ
1__inference_module_wrapper_9_layer_call_fn_389034eHвE
.в+
)К&
args_0         А
к

trainingp"К         А$╩
J__inference_module_wrapper_layer_call_and_return_conditional_losses_388762|fgGвD
-в*
(К%
args_0         00
к

trainingp "-в*
#К 
0         00@
Ъ ╩
J__inference_module_wrapper_layer_call_and_return_conditional_losses_388773|fgGвD
-в*
(К%
args_0         00
к

trainingp"-в*
#К 
0         00@
Ъ в
/__inference_module_wrapper_layer_call_fn_388782ofgGвD
-в*
(К%
args_0         00
к

trainingp " К         00@в
/__inference_module_wrapper_layer_call_fn_388791ofgGвD
-в*
(К%
args_0         00
к

trainingp" К         00@╧
F__inference_sequential_layer_call_and_return_conditional_losses_388429ДfghijklmnopqMвJ
Cв@
6К3
module_wrapper_input         00
p 

 
к "%в"
К
0         
Ъ ╧
F__inference_sequential_layer_call_and_return_conditional_losses_388472ДfghijklmnopqMвJ
Cв@
6К3
module_wrapper_input         00
p

 
к "%в"
К
0         
Ъ └
F__inference_sequential_layer_call_and_return_conditional_losses_388602vfghijklmnopq?в<
5в2
(К%
inputs         00
p 

 
к "%в"
К
0         
Ъ └
F__inference_sequential_layer_call_and_return_conditional_losses_388693vfghijklmnopq?в<
5в2
(К%
inputs         00
p

 
к "%в"
К
0         
Ъ ж
+__inference_sequential_layer_call_fn_387896wfghijklmnopqMвJ
Cв@
6К3
module_wrapper_input         00
p 

 
к "К         ж
+__inference_sequential_layer_call_fn_388386wfghijklmnopqMвJ
Cв@
6К3
module_wrapper_input         00
p

 
к "К         Ш
+__inference_sequential_layer_call_fn_388722ifghijklmnopq?в<
5в2
(К%
inputs         00
p 

 
к "К         Ш
+__inference_sequential_layer_call_fn_388751ifghijklmnopq?в<
5в2
(К%
inputs         00
p

 
к "К         ▌
$__inference_signature_wrapper_388509┤fghijklmnopq]вZ
в 
SкP
N
module_wrapper_input6К3
module_wrapper_input         00"EкB
@
module_wrapper_14+К(
module_wrapper_14         