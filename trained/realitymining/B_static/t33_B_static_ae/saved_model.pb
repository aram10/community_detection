â

Í
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718Ó	
z
dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ * 
shared_namedense_64/kernel
s
#dense_64/kernel/Read/ReadVariableOpReadVariableOpdense_64/kernel*
_output_shapes

:^ *
dtype0
r
dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_64/bias
k
!dense_64/bias/Read/ReadVariableOpReadVariableOpdense_64/bias*
_output_shapes
: *
dtype0
z
dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^* 
shared_namedense_65/kernel
s
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel*
_output_shapes

: ^*
dtype0
r
dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_65/bias
k
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
_output_shapes
:^*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ê
valueÀB½ B¶

history
encoder
decoder
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
y
	layer_with_weights-0
	layer-0

trainable_variables
	variables
regularization_losses
	keras_api
y
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api

0
1
2
3

0
1
2
3
 
­
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables

layers
	variables
layer_metrics
regularization_losses
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1

0
1
 
­
 layer_regularization_losses
!non_trainable_variables
"metrics

trainable_variables

#layers
	variables
$layer_metrics
regularization_losses
h

kernel
bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api

0
1

0
1
 
­
)layer_regularization_losses
*non_trainable_variables
+metrics
trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
US
VARIABLE_VALUEdense_64/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_64/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_65/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_65/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
 

0
1

0
1
 
­
.layer_regularization_losses
/metrics
0non_trainable_variables
trainable_variables

1layers
	variables
2layer_metrics
regularization_losses
 
 
 

	0
 

0
1

0
1
 
­
3layer_regularization_losses
4metrics
5non_trainable_variables
%trainable_variables

6layers
&	variables
7layer_metrics
'regularization_losses
 
 
 

0
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
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ^
ü
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_64/kerneldense_64/biasdense_65/kerneldense_65/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_16616249
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
±
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_64/kernel/Read/ReadVariableOp!dense_64/bias/Read/ReadVariableOp#dense_65/kernel/Read/ReadVariableOp!dense_65/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_16616755
Ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_64/kerneldense_64/biasdense_65/kerneldense_65/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_16616777¥ô
Ç
Ü
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616623
dense_65_input9
'dense_65_matmul_readvariableop_resource: ^6
(dense_65_biasadd_readvariableop_resource:^
identity¢dense_65/BiasAdd/ReadVariableOp¢dense_65/MatMul/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¨
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_65/MatMul/ReadVariableOp
dense_65/MatMulMatMuldense_65_input&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/MatMul§
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_65/BiasAdd/ReadVariableOp¥
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/BiasAdd|
dense_65/SigmoidSigmoiddense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/SigmoidÎ
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulß
IdentityIdentitydense_65/Sigmoid:y:0 ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_65_input
ä
³
__inference_loss_fn_1_16616703L
:dense_65_kernel_regularizer_square_readvariableop_resource: ^
identity¢1dense_65/kernel/Regularizer/Square/ReadVariableOpá
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_65_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mul
IdentityIdentity#dense_65/kernel/Regularizer/mul:z:02^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp
³
R
2__inference_dense_64_activity_regularizer_16615748

activation
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesd
MeanMean
activationMean/reduction_indices:output:0*
T0*
_output_shapes
:2
Mean[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2
	Maximum/yc
MaximumMaximumMean:output:0Maximum/y:output:0*
T0*
_output_shapes
:2	
Maximum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
	truediv/xa
truedivRealDivtruediv/x:output:0Maximum:z:0*
T0*
_output_shapes
:2	
truedivA
LogLogtruediv:z:0*
T0*
_output_shapes
:2
LogS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
mul/xM
mulMulmul/x:output:0Log:y:0*
T0*
_output_shapes
:2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Maximum:z:0*
T0*
_output_shapes
:2
sub_
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2
truediv_1/xc
	truediv_1RealDivtruediv_1/x:output:0sub:z:0*
T0*
_output_shapes
:2
	truediv_1G
Log_1Logtruediv_1:z:0*
T0*
_output_shapes
:2
Log_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2	
mul_1/xU
mul_1Mulmul_1/x:output:0	Log_1:y:0*
T0*
_output_shapes
:2
mul_1J
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add>
RankRankadd:z:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rangeK
SumSumadd:z:0range:output:0*
T0*
_output_shapes
: 2
SumW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
mul_2/xV
mul_2Mulmul_2/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mul_2L
IdentityIdentity	mul_2:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::D @

_output_shapes
:
$
_user_specified_name
activation
Ç
Ü
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616606
dense_65_input9
'dense_65_matmul_readvariableop_resource: ^6
(dense_65_biasadd_readvariableop_resource:^
identity¢dense_65/BiasAdd/ReadVariableOp¢dense_65/MatMul/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¨
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_65/MatMul/ReadVariableOp
dense_65/MatMulMatMuldense_65_input&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/MatMul§
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_65/BiasAdd/ReadVariableOp¥
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/BiasAdd|
dense_65/SigmoidSigmoiddense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/SigmoidÎ
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulß
IdentityIdentitydense_65/Sigmoid:y:0 ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_65_input
²A
ä
K__inference_sequential_64_layer_call_and_return_conditional_losses_16616513

inputs9
'dense_64_matmul_readvariableop_resource:^ 6
(dense_64_biasadd_readvariableop_resource: 
identity

identity_1¢dense_64/BiasAdd/ReadVariableOp¢dense_64/MatMul/ReadVariableOp¢1dense_64/kernel/Regularizer/Square/ReadVariableOp¨
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02 
dense_64/MatMul/ReadVariableOp
dense_64/MatMulMatMulinputs&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_64/MatMul§
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_64/BiasAdd/ReadVariableOp¥
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_64/BiasAdd|
dense_64/SigmoidSigmoiddense_64/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_64/Sigmoid¬
3dense_64/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_64/ActivityRegularizer/Mean/reduction_indicesÇ
!dense_64/ActivityRegularizer/MeanMeandense_64/Sigmoid:y:0<dense_64/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2#
!dense_64/ActivityRegularizer/Mean
&dense_64/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2(
&dense_64/ActivityRegularizer/Maximum/yÙ
$dense_64/ActivityRegularizer/MaximumMaximum*dense_64/ActivityRegularizer/Mean:output:0/dense_64/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2&
$dense_64/ActivityRegularizer/Maximum
&dense_64/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&dense_64/ActivityRegularizer/truediv/x×
$dense_64/ActivityRegularizer/truedivRealDiv/dense_64/ActivityRegularizer/truediv/x:output:0(dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2&
$dense_64/ActivityRegularizer/truediv
 dense_64/ActivityRegularizer/LogLog(dense_64/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/Log
"dense_64/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"dense_64/ActivityRegularizer/mul/xÃ
 dense_64/ActivityRegularizer/mulMul+dense_64/ActivityRegularizer/mul/x:output:0$dense_64/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/mul
"dense_64/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dense_64/ActivityRegularizer/sub/xÇ
 dense_64/ActivityRegularizer/subSub+dense_64/ActivityRegularizer/sub/x:output:0(dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/sub
(dense_64/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2*
(dense_64/ActivityRegularizer/truediv_1/xÙ
&dense_64/ActivityRegularizer/truediv_1RealDiv1dense_64/ActivityRegularizer/truediv_1/x:output:0$dense_64/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2(
&dense_64/ActivityRegularizer/truediv_1 
"dense_64/ActivityRegularizer/Log_1Log*dense_64/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2$
"dense_64/ActivityRegularizer/Log_1
$dense_64/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2&
$dense_64/ActivityRegularizer/mul_1/xË
"dense_64/ActivityRegularizer/mul_1Mul-dense_64/ActivityRegularizer/mul_1/x:output:0&dense_64/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2$
"dense_64/ActivityRegularizer/mul_1À
 dense_64/ActivityRegularizer/addAddV2$dense_64/ActivityRegularizer/mul:z:0&dense_64/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/add
"dense_64/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_64/ActivityRegularizer/Const¿
 dense_64/ActivityRegularizer/SumSum$dense_64/ActivityRegularizer/add:z:0+dense_64/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/Sum
$dense_64/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dense_64/ActivityRegularizer/mul_2/xÊ
"dense_64/ActivityRegularizer/mul_2Mul-dense_64/ActivityRegularizer/mul_2/x:output:0)dense_64/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_64/ActivityRegularizer/mul_2
"dense_64/ActivityRegularizer/ShapeShapedense_64/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_64/ActivityRegularizer/Shape®
0dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_64/ActivityRegularizer/strided_slice/stack²
2dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_1²
2dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_2
*dense_64/ActivityRegularizer/strided_sliceStridedSlice+dense_64/ActivityRegularizer/Shape:output:09dense_64/ActivityRegularizer/strided_slice/stack:output:0;dense_64/ActivityRegularizer/strided_slice/stack_1:output:0;dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_64/ActivityRegularizer/strided_slice³
!dense_64/ActivityRegularizer/CastCast3dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_64/ActivityRegularizer/CastË
&dense_64/ActivityRegularizer/truediv_2RealDiv&dense_64/ActivityRegularizer/mul_2:z:0%dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_64/ActivityRegularizer/truediv_2Î
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulß
IdentityIdentitydense_64/Sigmoid:y:0 ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityè

Identity_1Identity*dense_64/ActivityRegularizer/truediv_2:z:0 ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs


K__inference_sequential_65_layer_call_and_return_conditional_losses_16616006

inputs#
dense_65_16615994: ^
dense_65_16615996:^
identity¢ dense_65/StatefulPartitionedCall¢1dense_65/kernel/Regularizer/Square/ReadVariableOp
 dense_65/StatefulPartitionedCallStatefulPartitionedCallinputsdense_65_16615994dense_65_16615996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_166159502"
 dense_65/StatefulPartitionedCall¸
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_65_16615994*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulÔ
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0!^dense_65/StatefulPartitionedCall2^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
²

0__inference_sequential_64_layer_call_fn_16615878
input_33
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_33unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_64_layer_call_and_return_conditional_losses_166158602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_33
¬

0__inference_sequential_64_layer_call_fn_16616411

inputs
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_64_layer_call_and_return_conditional_losses_166157942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
%
Ô
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616222
input_1(
sequential_64_16616197:^ $
sequential_64_16616199: (
sequential_65_16616203: ^$
sequential_65_16616205:^
identity

identity_1¢1dense_64/kernel/Regularizer/Square/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¢%sequential_64/StatefulPartitionedCall¢%sequential_65/StatefulPartitionedCall·
%sequential_64/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_64_16616197sequential_64_16616199*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_64_layer_call_and_return_conditional_losses_166158602'
%sequential_64/StatefulPartitionedCallÛ
%sequential_65/StatefulPartitionedCallStatefulPartitionedCall.sequential_64/StatefulPartitionedCall:output:0sequential_65_16616203sequential_65_16616205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_65_layer_call_and_return_conditional_losses_166160062'
%sequential_65/StatefulPartitionedCall½
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_64_16616197*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mul½
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_65_16616203*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulº
IdentityIdentity.sequential_65/StatefulPartitionedCall:output:02^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp&^sequential_64/StatefulPartitionedCall&^sequential_65/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_64/StatefulPartitionedCall:output:12^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp&^sequential_64/StatefulPartitionedCall&^sequential_65/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_64/StatefulPartitionedCall%sequential_64/StatefulPartitionedCall2N
%sequential_65/StatefulPartitionedCall%sequential_65/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
²A
ä
K__inference_sequential_64_layer_call_and_return_conditional_losses_16616467

inputs9
'dense_64_matmul_readvariableop_resource:^ 6
(dense_64_biasadd_readvariableop_resource: 
identity

identity_1¢dense_64/BiasAdd/ReadVariableOp¢dense_64/MatMul/ReadVariableOp¢1dense_64/kernel/Regularizer/Square/ReadVariableOp¨
dense_64/MatMul/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02 
dense_64/MatMul/ReadVariableOp
dense_64/MatMulMatMulinputs&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_64/MatMul§
dense_64/BiasAdd/ReadVariableOpReadVariableOp(dense_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_64/BiasAdd/ReadVariableOp¥
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_64/BiasAdd|
dense_64/SigmoidSigmoiddense_64/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_64/Sigmoid¬
3dense_64/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_64/ActivityRegularizer/Mean/reduction_indicesÇ
!dense_64/ActivityRegularizer/MeanMeandense_64/Sigmoid:y:0<dense_64/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2#
!dense_64/ActivityRegularizer/Mean
&dense_64/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2(
&dense_64/ActivityRegularizer/Maximum/yÙ
$dense_64/ActivityRegularizer/MaximumMaximum*dense_64/ActivityRegularizer/Mean:output:0/dense_64/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2&
$dense_64/ActivityRegularizer/Maximum
&dense_64/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&dense_64/ActivityRegularizer/truediv/x×
$dense_64/ActivityRegularizer/truedivRealDiv/dense_64/ActivityRegularizer/truediv/x:output:0(dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2&
$dense_64/ActivityRegularizer/truediv
 dense_64/ActivityRegularizer/LogLog(dense_64/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/Log
"dense_64/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"dense_64/ActivityRegularizer/mul/xÃ
 dense_64/ActivityRegularizer/mulMul+dense_64/ActivityRegularizer/mul/x:output:0$dense_64/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/mul
"dense_64/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dense_64/ActivityRegularizer/sub/xÇ
 dense_64/ActivityRegularizer/subSub+dense_64/ActivityRegularizer/sub/x:output:0(dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/sub
(dense_64/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2*
(dense_64/ActivityRegularizer/truediv_1/xÙ
&dense_64/ActivityRegularizer/truediv_1RealDiv1dense_64/ActivityRegularizer/truediv_1/x:output:0$dense_64/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2(
&dense_64/ActivityRegularizer/truediv_1 
"dense_64/ActivityRegularizer/Log_1Log*dense_64/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2$
"dense_64/ActivityRegularizer/Log_1
$dense_64/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2&
$dense_64/ActivityRegularizer/mul_1/xË
"dense_64/ActivityRegularizer/mul_1Mul-dense_64/ActivityRegularizer/mul_1/x:output:0&dense_64/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2$
"dense_64/ActivityRegularizer/mul_1À
 dense_64/ActivityRegularizer/addAddV2$dense_64/ActivityRegularizer/mul:z:0&dense_64/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/add
"dense_64/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_64/ActivityRegularizer/Const¿
 dense_64/ActivityRegularizer/SumSum$dense_64/ActivityRegularizer/add:z:0+dense_64/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_64/ActivityRegularizer/Sum
$dense_64/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dense_64/ActivityRegularizer/mul_2/xÊ
"dense_64/ActivityRegularizer/mul_2Mul-dense_64/ActivityRegularizer/mul_2/x:output:0)dense_64/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_64/ActivityRegularizer/mul_2
"dense_64/ActivityRegularizer/ShapeShapedense_64/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_64/ActivityRegularizer/Shape®
0dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_64/ActivityRegularizer/strided_slice/stack²
2dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_1²
2dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_2
*dense_64/ActivityRegularizer/strided_sliceStridedSlice+dense_64/ActivityRegularizer/Shape:output:09dense_64/ActivityRegularizer/strided_slice/stack:output:0;dense_64/ActivityRegularizer/strided_slice/stack_1:output:0;dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_64/ActivityRegularizer/strided_slice³
!dense_64/ActivityRegularizer/CastCast3dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_64/ActivityRegularizer/CastË
&dense_64/ActivityRegularizer/truediv_2RealDiv&dense_64/ActivityRegularizer/mul_2:z:0%dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_64/ActivityRegularizer/truediv_2Î
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulß
IdentityIdentitydense_64/Sigmoid:y:0 ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityè

Identity_1Identity*dense_64/ActivityRegularizer/truediv_2:z:0 ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
Î
Ê
&__inference_signature_wrapper_16616249
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_166157192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
©

0__inference_sequential_65_layer_call_fn_16616537

inputs
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_65_layer_call_and_return_conditional_losses_166159632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­
«
F__inference_dense_65_layer_call_and_return_conditional_losses_16615950

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2	
SigmoidÅ
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó
Ï
1__inference_autoencoder_32_layer_call_fn_16616277
x
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_166161402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
Êe
º
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616336
xG
5sequential_64_dense_64_matmul_readvariableop_resource:^ D
6sequential_64_dense_64_biasadd_readvariableop_resource: G
5sequential_65_dense_65_matmul_readvariableop_resource: ^D
6sequential_65_dense_65_biasadd_readvariableop_resource:^
identity

identity_1¢1dense_64/kernel/Regularizer/Square/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¢-sequential_64/dense_64/BiasAdd/ReadVariableOp¢,sequential_64/dense_64/MatMul/ReadVariableOp¢-sequential_65/dense_65/BiasAdd/ReadVariableOp¢,sequential_65/dense_65/MatMul/ReadVariableOpÒ
,sequential_64/dense_64/MatMul/ReadVariableOpReadVariableOp5sequential_64_dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02.
,sequential_64/dense_64/MatMul/ReadVariableOp³
sequential_64/dense_64/MatMulMatMulx4sequential_64/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_64/dense_64/MatMulÑ
-sequential_64/dense_64/BiasAdd/ReadVariableOpReadVariableOp6sequential_64_dense_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_64/dense_64/BiasAdd/ReadVariableOpÝ
sequential_64/dense_64/BiasAddBiasAdd'sequential_64/dense_64/MatMul:product:05sequential_64/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_64/dense_64/BiasAdd¦
sequential_64/dense_64/SigmoidSigmoid'sequential_64/dense_64/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_64/dense_64/SigmoidÈ
Asequential_64/dense_64/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_64/dense_64/ActivityRegularizer/Mean/reduction_indicesÿ
/sequential_64/dense_64/ActivityRegularizer/MeanMean"sequential_64/dense_64/Sigmoid:y:0Jsequential_64/dense_64/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 21
/sequential_64/dense_64/ActivityRegularizer/Mean±
4sequential_64/dense_64/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.26
4sequential_64/dense_64/ActivityRegularizer/Maximum/y
2sequential_64/dense_64/ActivityRegularizer/MaximumMaximum8sequential_64/dense_64/ActivityRegularizer/Mean:output:0=sequential_64/dense_64/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 24
2sequential_64/dense_64/ActivityRegularizer/Maximum±
4sequential_64/dense_64/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<26
4sequential_64/dense_64/ActivityRegularizer/truediv/x
2sequential_64/dense_64/ActivityRegularizer/truedivRealDiv=sequential_64/dense_64/ActivityRegularizer/truediv/x:output:06sequential_64/dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 24
2sequential_64/dense_64/ActivityRegularizer/truedivÄ
.sequential_64/dense_64/ActivityRegularizer/LogLog6sequential_64/dense_64/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/Log©
0sequential_64/dense_64/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<22
0sequential_64/dense_64/ActivityRegularizer/mul/xû
.sequential_64/dense_64/ActivityRegularizer/mulMul9sequential_64/dense_64/ActivityRegularizer/mul/x:output:02sequential_64/dense_64/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/mul©
0sequential_64/dense_64/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_64/dense_64/ActivityRegularizer/sub/xÿ
.sequential_64/dense_64/ActivityRegularizer/subSub9sequential_64/dense_64/ActivityRegularizer/sub/x:output:06sequential_64/dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/subµ
6sequential_64/dense_64/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?28
6sequential_64/dense_64/ActivityRegularizer/truediv_1/x
4sequential_64/dense_64/ActivityRegularizer/truediv_1RealDiv?sequential_64/dense_64/ActivityRegularizer/truediv_1/x:output:02sequential_64/dense_64/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 26
4sequential_64/dense_64/ActivityRegularizer/truediv_1Ê
0sequential_64/dense_64/ActivityRegularizer/Log_1Log8sequential_64/dense_64/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 22
0sequential_64/dense_64/ActivityRegularizer/Log_1­
2sequential_64/dense_64/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?24
2sequential_64/dense_64/ActivityRegularizer/mul_1/x
0sequential_64/dense_64/ActivityRegularizer/mul_1Mul;sequential_64/dense_64/ActivityRegularizer/mul_1/x:output:04sequential_64/dense_64/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 22
0sequential_64/dense_64/ActivityRegularizer/mul_1ø
.sequential_64/dense_64/ActivityRegularizer/addAddV22sequential_64/dense_64/ActivityRegularizer/mul:z:04sequential_64/dense_64/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/add®
0sequential_64/dense_64/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_64/dense_64/ActivityRegularizer/Const÷
.sequential_64/dense_64/ActivityRegularizer/SumSum2sequential_64/dense_64/ActivityRegularizer/add:z:09sequential_64/dense_64/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/Sum­
2sequential_64/dense_64/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_64/dense_64/ActivityRegularizer/mul_2/x
0sequential_64/dense_64/ActivityRegularizer/mul_2Mul;sequential_64/dense_64/ActivityRegularizer/mul_2/x:output:07sequential_64/dense_64/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_64/dense_64/ActivityRegularizer/mul_2¶
0sequential_64/dense_64/ActivityRegularizer/ShapeShape"sequential_64/dense_64/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_64/dense_64/ActivityRegularizer/ShapeÊ
>sequential_64/dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_64/dense_64/ActivityRegularizer/strided_slice/stackÎ
@sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1Î
@sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2ä
8sequential_64/dense_64/ActivityRegularizer/strided_sliceStridedSlice9sequential_64/dense_64/ActivityRegularizer/Shape:output:0Gsequential_64/dense_64/ActivityRegularizer/strided_slice/stack:output:0Isequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_64/dense_64/ActivityRegularizer/strided_sliceÝ
/sequential_64/dense_64/ActivityRegularizer/CastCastAsequential_64/dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_64/dense_64/ActivityRegularizer/Cast
4sequential_64/dense_64/ActivityRegularizer/truediv_2RealDiv4sequential_64/dense_64/ActivityRegularizer/mul_2:z:03sequential_64/dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_64/dense_64/ActivityRegularizer/truediv_2Ò
,sequential_65/dense_65/MatMul/ReadVariableOpReadVariableOp5sequential_65_dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02.
,sequential_65/dense_65/MatMul/ReadVariableOpÔ
sequential_65/dense_65/MatMulMatMul"sequential_64/dense_64/Sigmoid:y:04sequential_65/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
sequential_65/dense_65/MatMulÑ
-sequential_65/dense_65/BiasAdd/ReadVariableOpReadVariableOp6sequential_65_dense_65_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02/
-sequential_65/dense_65/BiasAdd/ReadVariableOpÝ
sequential_65/dense_65/BiasAddBiasAdd'sequential_65/dense_65/MatMul:product:05sequential_65/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_65/dense_65/BiasAdd¦
sequential_65/dense_65/SigmoidSigmoid'sequential_65/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_65/dense_65/SigmoidÜ
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_64_dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulÜ
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_65_dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mul
IdentityIdentity"sequential_65/dense_65/Sigmoid:y:02^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp.^sequential_64/dense_64/BiasAdd/ReadVariableOp-^sequential_64/dense_64/MatMul/ReadVariableOp.^sequential_65/dense_65/BiasAdd/ReadVariableOp-^sequential_65/dense_65/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity¥

Identity_1Identity8sequential_64/dense_64/ActivityRegularizer/truediv_2:z:02^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp.^sequential_64/dense_64/BiasAdd/ReadVariableOp-^sequential_64/dense_64/MatMul/ReadVariableOp.^sequential_65/dense_65/BiasAdd/ReadVariableOp-^sequential_65/dense_65/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_64/dense_64/BiasAdd/ReadVariableOp-sequential_64/dense_64/BiasAdd/ReadVariableOp2\
,sequential_64/dense_64/MatMul/ReadVariableOp,sequential_64/dense_64/MatMul/ReadVariableOp2^
-sequential_65/dense_65/BiasAdd/ReadVariableOp-sequential_65/dense_65/BiasAdd/ReadVariableOp2\
,sequential_65/dense_65/MatMul/ReadVariableOp,sequential_65/dense_65/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
ô"

K__inference_sequential_64_layer_call_and_return_conditional_losses_16615902
input_33#
dense_64_16615881:^ 
dense_64_16615883: 
identity

identity_1¢ dense_64/StatefulPartitionedCall¢1dense_64/kernel/Regularizer/Square/ReadVariableOp
 dense_64/StatefulPartitionedCallStatefulPartitionedCallinput_33dense_64_16615881dense_64_16615883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_64_layer_call_and_return_conditional_losses_166157722"
 dense_64/StatefulPartitionedCallü
,dense_64/ActivityRegularizer/PartitionedCallPartitionedCall)dense_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_dense_64_activity_regularizer_166157482.
,dense_64/ActivityRegularizer/PartitionedCall¡
"dense_64/ActivityRegularizer/ShapeShape)dense_64/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_64/ActivityRegularizer/Shape®
0dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_64/ActivityRegularizer/strided_slice/stack²
2dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_1²
2dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_2
*dense_64/ActivityRegularizer/strided_sliceStridedSlice+dense_64/ActivityRegularizer/Shape:output:09dense_64/ActivityRegularizer/strided_slice/stack:output:0;dense_64/ActivityRegularizer/strided_slice/stack_1:output:0;dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_64/ActivityRegularizer/strided_slice³
!dense_64/ActivityRegularizer/CastCast3dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_64/ActivityRegularizer/CastÖ
$dense_64/ActivityRegularizer/truedivRealDiv5dense_64/ActivityRegularizer/PartitionedCall:output:0%dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_64/ActivityRegularizer/truediv¸
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_64_16615881*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulÔ
IdentityIdentity)dense_64/StatefulPartitionedCall:output:0!^dense_64/StatefulPartitionedCall2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_64/ActivityRegularizer/truediv:z:0!^dense_64/StatefulPartitionedCall2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_33
Êe
º
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616395
xG
5sequential_64_dense_64_matmul_readvariableop_resource:^ D
6sequential_64_dense_64_biasadd_readvariableop_resource: G
5sequential_65_dense_65_matmul_readvariableop_resource: ^D
6sequential_65_dense_65_biasadd_readvariableop_resource:^
identity

identity_1¢1dense_64/kernel/Regularizer/Square/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¢-sequential_64/dense_64/BiasAdd/ReadVariableOp¢,sequential_64/dense_64/MatMul/ReadVariableOp¢-sequential_65/dense_65/BiasAdd/ReadVariableOp¢,sequential_65/dense_65/MatMul/ReadVariableOpÒ
,sequential_64/dense_64/MatMul/ReadVariableOpReadVariableOp5sequential_64_dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02.
,sequential_64/dense_64/MatMul/ReadVariableOp³
sequential_64/dense_64/MatMulMatMulx4sequential_64/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_64/dense_64/MatMulÑ
-sequential_64/dense_64/BiasAdd/ReadVariableOpReadVariableOp6sequential_64_dense_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_64/dense_64/BiasAdd/ReadVariableOpÝ
sequential_64/dense_64/BiasAddBiasAdd'sequential_64/dense_64/MatMul:product:05sequential_64/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_64/dense_64/BiasAdd¦
sequential_64/dense_64/SigmoidSigmoid'sequential_64/dense_64/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_64/dense_64/SigmoidÈ
Asequential_64/dense_64/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_64/dense_64/ActivityRegularizer/Mean/reduction_indicesÿ
/sequential_64/dense_64/ActivityRegularizer/MeanMean"sequential_64/dense_64/Sigmoid:y:0Jsequential_64/dense_64/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 21
/sequential_64/dense_64/ActivityRegularizer/Mean±
4sequential_64/dense_64/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.26
4sequential_64/dense_64/ActivityRegularizer/Maximum/y
2sequential_64/dense_64/ActivityRegularizer/MaximumMaximum8sequential_64/dense_64/ActivityRegularizer/Mean:output:0=sequential_64/dense_64/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 24
2sequential_64/dense_64/ActivityRegularizer/Maximum±
4sequential_64/dense_64/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<26
4sequential_64/dense_64/ActivityRegularizer/truediv/x
2sequential_64/dense_64/ActivityRegularizer/truedivRealDiv=sequential_64/dense_64/ActivityRegularizer/truediv/x:output:06sequential_64/dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 24
2sequential_64/dense_64/ActivityRegularizer/truedivÄ
.sequential_64/dense_64/ActivityRegularizer/LogLog6sequential_64/dense_64/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/Log©
0sequential_64/dense_64/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<22
0sequential_64/dense_64/ActivityRegularizer/mul/xû
.sequential_64/dense_64/ActivityRegularizer/mulMul9sequential_64/dense_64/ActivityRegularizer/mul/x:output:02sequential_64/dense_64/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/mul©
0sequential_64/dense_64/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_64/dense_64/ActivityRegularizer/sub/xÿ
.sequential_64/dense_64/ActivityRegularizer/subSub9sequential_64/dense_64/ActivityRegularizer/sub/x:output:06sequential_64/dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/subµ
6sequential_64/dense_64/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?28
6sequential_64/dense_64/ActivityRegularizer/truediv_1/x
4sequential_64/dense_64/ActivityRegularizer/truediv_1RealDiv?sequential_64/dense_64/ActivityRegularizer/truediv_1/x:output:02sequential_64/dense_64/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 26
4sequential_64/dense_64/ActivityRegularizer/truediv_1Ê
0sequential_64/dense_64/ActivityRegularizer/Log_1Log8sequential_64/dense_64/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 22
0sequential_64/dense_64/ActivityRegularizer/Log_1­
2sequential_64/dense_64/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?24
2sequential_64/dense_64/ActivityRegularizer/mul_1/x
0sequential_64/dense_64/ActivityRegularizer/mul_1Mul;sequential_64/dense_64/ActivityRegularizer/mul_1/x:output:04sequential_64/dense_64/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 22
0sequential_64/dense_64/ActivityRegularizer/mul_1ø
.sequential_64/dense_64/ActivityRegularizer/addAddV22sequential_64/dense_64/ActivityRegularizer/mul:z:04sequential_64/dense_64/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/add®
0sequential_64/dense_64/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_64/dense_64/ActivityRegularizer/Const÷
.sequential_64/dense_64/ActivityRegularizer/SumSum2sequential_64/dense_64/ActivityRegularizer/add:z:09sequential_64/dense_64/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_64/dense_64/ActivityRegularizer/Sum­
2sequential_64/dense_64/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_64/dense_64/ActivityRegularizer/mul_2/x
0sequential_64/dense_64/ActivityRegularizer/mul_2Mul;sequential_64/dense_64/ActivityRegularizer/mul_2/x:output:07sequential_64/dense_64/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_64/dense_64/ActivityRegularizer/mul_2¶
0sequential_64/dense_64/ActivityRegularizer/ShapeShape"sequential_64/dense_64/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_64/dense_64/ActivityRegularizer/ShapeÊ
>sequential_64/dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_64/dense_64/ActivityRegularizer/strided_slice/stackÎ
@sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1Î
@sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2ä
8sequential_64/dense_64/ActivityRegularizer/strided_sliceStridedSlice9sequential_64/dense_64/ActivityRegularizer/Shape:output:0Gsequential_64/dense_64/ActivityRegularizer/strided_slice/stack:output:0Isequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_64/dense_64/ActivityRegularizer/strided_sliceÝ
/sequential_64/dense_64/ActivityRegularizer/CastCastAsequential_64/dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_64/dense_64/ActivityRegularizer/Cast
4sequential_64/dense_64/ActivityRegularizer/truediv_2RealDiv4sequential_64/dense_64/ActivityRegularizer/mul_2:z:03sequential_64/dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_64/dense_64/ActivityRegularizer/truediv_2Ò
,sequential_65/dense_65/MatMul/ReadVariableOpReadVariableOp5sequential_65_dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02.
,sequential_65/dense_65/MatMul/ReadVariableOpÔ
sequential_65/dense_65/MatMulMatMul"sequential_64/dense_64/Sigmoid:y:04sequential_65/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
sequential_65/dense_65/MatMulÑ
-sequential_65/dense_65/BiasAdd/ReadVariableOpReadVariableOp6sequential_65_dense_65_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02/
-sequential_65/dense_65/BiasAdd/ReadVariableOpÝ
sequential_65/dense_65/BiasAddBiasAdd'sequential_65/dense_65/MatMul:product:05sequential_65/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_65/dense_65/BiasAdd¦
sequential_65/dense_65/SigmoidSigmoid'sequential_65/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_65/dense_65/SigmoidÜ
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_64_dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulÜ
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_65_dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mul
IdentityIdentity"sequential_65/dense_65/Sigmoid:y:02^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp.^sequential_64/dense_64/BiasAdd/ReadVariableOp-^sequential_64/dense_64/MatMul/ReadVariableOp.^sequential_65/dense_65/BiasAdd/ReadVariableOp-^sequential_65/dense_65/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity¥

Identity_1Identity8sequential_64/dense_64/ActivityRegularizer/truediv_2:z:02^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp.^sequential_64/dense_64/BiasAdd/ReadVariableOp-^sequential_64/dense_64/MatMul/ReadVariableOp.^sequential_65/dense_65/BiasAdd/ReadVariableOp-^sequential_65/dense_65/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_64/dense_64/BiasAdd/ReadVariableOp-sequential_64/dense_64/BiasAdd/ReadVariableOp2\
,sequential_64/dense_64/MatMul/ReadVariableOp,sequential_64/dense_64/MatMul/ReadVariableOp2^
-sequential_65/dense_65/BiasAdd/ReadVariableOp-sequential_65/dense_65/BiasAdd/ReadVariableOp2\
,sequential_65/dense_65/MatMul/ReadVariableOp,sequential_65/dense_65/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
¯
Ô
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616572

inputs9
'dense_65_matmul_readvariableop_resource: ^6
(dense_65_biasadd_readvariableop_resource:^
identity¢dense_65/BiasAdd/ReadVariableOp¢dense_65/MatMul/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¨
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_65/MatMul/ReadVariableOp
dense_65/MatMulMatMulinputs&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/MatMul§
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_65/BiasAdd/ReadVariableOp¥
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/BiasAdd|
dense_65/SigmoidSigmoiddense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/SigmoidÎ
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulß
IdentityIdentitydense_65/Sigmoid:y:0 ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ä]

#__inference__wrapped_model_16615719
input_1V
Dautoencoder_32_sequential_64_dense_64_matmul_readvariableop_resource:^ S
Eautoencoder_32_sequential_64_dense_64_biasadd_readvariableop_resource: V
Dautoencoder_32_sequential_65_dense_65_matmul_readvariableop_resource: ^S
Eautoencoder_32_sequential_65_dense_65_biasadd_readvariableop_resource:^
identity¢<autoencoder_32/sequential_64/dense_64/BiasAdd/ReadVariableOp¢;autoencoder_32/sequential_64/dense_64/MatMul/ReadVariableOp¢<autoencoder_32/sequential_65/dense_65/BiasAdd/ReadVariableOp¢;autoencoder_32/sequential_65/dense_65/MatMul/ReadVariableOpÿ
;autoencoder_32/sequential_64/dense_64/MatMul/ReadVariableOpReadVariableOpDautoencoder_32_sequential_64_dense_64_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02=
;autoencoder_32/sequential_64/dense_64/MatMul/ReadVariableOpæ
,autoencoder_32/sequential_64/dense_64/MatMulMatMulinput_1Cautoencoder_32/sequential_64/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,autoencoder_32/sequential_64/dense_64/MatMulþ
<autoencoder_32/sequential_64/dense_64/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_32_sequential_64_dense_64_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<autoencoder_32/sequential_64/dense_64/BiasAdd/ReadVariableOp
-autoencoder_32/sequential_64/dense_64/BiasAddBiasAdd6autoencoder_32/sequential_64/dense_64/MatMul:product:0Dautoencoder_32/sequential_64/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-autoencoder_32/sequential_64/dense_64/BiasAddÓ
-autoencoder_32/sequential_64/dense_64/SigmoidSigmoid6autoencoder_32/sequential_64/dense_64/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-autoencoder_32/sequential_64/dense_64/Sigmoidæ
Pautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2R
Pautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Mean/reduction_indices»
>autoencoder_32/sequential_64/dense_64/ActivityRegularizer/MeanMean1autoencoder_32/sequential_64/dense_64/Sigmoid:y:0Yautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2@
>autoencoder_32/sequential_64/dense_64/ActivityRegularizer/MeanÏ
Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2E
Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Maximum/yÍ
Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/MaximumMaximumGautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Mean:output:0Lautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/MaximumÏ
Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2E
Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv/xË
Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truedivRealDivLautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv/x:output:0Eautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truedivñ
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/LogLogEautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2?
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/LogÇ
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2A
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul/x·
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/mulMulHautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul/x:output:0Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2?
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/mulÇ
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/sub/x»
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/subSubHautoencoder_32/sequential_64/dense_64/ActivityRegularizer/sub/x:output:0Eautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2?
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/subÓ
Eautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2G
Eautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv_1/xÍ
Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv_1RealDivNautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv_1/x:output:0Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv_1÷
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/Log_1LogGautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/Log_1Ë
Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2C
Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_1/x¿
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_1MulJautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_1/x:output:0Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2A
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_1´
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/addAddV2Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul:z:0Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2?
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/addÌ
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2A
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/Const³
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/SumSumAautoencoder_32/sequential_64/dense_64/ActivityRegularizer/add:z:0Hautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2?
=autoencoder_32/sequential_64/dense_64/ActivityRegularizer/SumË
Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2C
Aautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_2/x¾
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_2MulJautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_2/x:output:0Fautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_2ã
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/ShapeShape1autoencoder_32/sequential_64/dense_64/Sigmoid:y:0*
T0*
_output_shapes
:2A
?autoencoder_32/sequential_64/dense_64/ActivityRegularizer/Shapeè
Mautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stackì
Oautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1ì
Oautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2¾
Gautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_sliceStridedSliceHautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Shape:output:0Vautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stack:output:0Xautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_1:output:0Xautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice
>autoencoder_32/sequential_64/dense_64/ActivityRegularizer/CastCastPautoencoder_32/sequential_64/dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>autoencoder_32/sequential_64/dense_64/ActivityRegularizer/Cast¿
Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv_2RealDivCautoencoder_32/sequential_64/dense_64/ActivityRegularizer/mul_2:z:0Bautoencoder_32/sequential_64/dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2E
Cautoencoder_32/sequential_64/dense_64/ActivityRegularizer/truediv_2ÿ
;autoencoder_32/sequential_65/dense_65/MatMul/ReadVariableOpReadVariableOpDautoencoder_32_sequential_65_dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02=
;autoencoder_32/sequential_65/dense_65/MatMul/ReadVariableOp
,autoencoder_32/sequential_65/dense_65/MatMulMatMul1autoencoder_32/sequential_64/dense_64/Sigmoid:y:0Cautoencoder_32/sequential_65/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2.
,autoencoder_32/sequential_65/dense_65/MatMulþ
<autoencoder_32/sequential_65/dense_65/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_32_sequential_65_dense_65_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02>
<autoencoder_32/sequential_65/dense_65/BiasAdd/ReadVariableOp
-autoencoder_32/sequential_65/dense_65/BiasAddBiasAdd6autoencoder_32/sequential_65/dense_65/MatMul:product:0Dautoencoder_32/sequential_65/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2/
-autoencoder_32/sequential_65/dense_65/BiasAddÓ
-autoencoder_32/sequential_65/dense_65/SigmoidSigmoid6autoencoder_32/sequential_65/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2/
-autoencoder_32/sequential_65/dense_65/Sigmoidÿ
IdentityIdentity1autoencoder_32/sequential_65/dense_65/Sigmoid:y:0=^autoencoder_32/sequential_64/dense_64/BiasAdd/ReadVariableOp<^autoencoder_32/sequential_64/dense_64/MatMul/ReadVariableOp=^autoencoder_32/sequential_65/dense_65/BiasAdd/ReadVariableOp<^autoencoder_32/sequential_65/dense_65/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2|
<autoencoder_32/sequential_64/dense_64/BiasAdd/ReadVariableOp<autoencoder_32/sequential_64/dense_64/BiasAdd/ReadVariableOp2z
;autoencoder_32/sequential_64/dense_64/MatMul/ReadVariableOp;autoencoder_32/sequential_64/dense_64/MatMul/ReadVariableOp2|
<autoencoder_32/sequential_65/dense_65/BiasAdd/ReadVariableOp<autoencoder_32/sequential_65/dense_65/BiasAdd/ReadVariableOp2z
;autoencoder_32/sequential_65/dense_65/MatMul/ReadVariableOp;autoencoder_32/sequential_65/dense_65/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
ó
Ï
1__inference_autoencoder_32_layer_call_fn_16616263
x
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_166160842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
ô"

K__inference_sequential_64_layer_call_and_return_conditional_losses_16615926
input_33#
dense_64_16615905:^ 
dense_64_16615907: 
identity

identity_1¢ dense_64/StatefulPartitionedCall¢1dense_64/kernel/Regularizer/Square/ReadVariableOp
 dense_64/StatefulPartitionedCallStatefulPartitionedCallinput_33dense_64_16615905dense_64_16615907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_64_layer_call_and_return_conditional_losses_166157722"
 dense_64/StatefulPartitionedCallü
,dense_64/ActivityRegularizer/PartitionedCallPartitionedCall)dense_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_dense_64_activity_regularizer_166157482.
,dense_64/ActivityRegularizer/PartitionedCall¡
"dense_64/ActivityRegularizer/ShapeShape)dense_64/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_64/ActivityRegularizer/Shape®
0dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_64/ActivityRegularizer/strided_slice/stack²
2dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_1²
2dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_2
*dense_64/ActivityRegularizer/strided_sliceStridedSlice+dense_64/ActivityRegularizer/Shape:output:09dense_64/ActivityRegularizer/strided_slice/stack:output:0;dense_64/ActivityRegularizer/strided_slice/stack_1:output:0;dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_64/ActivityRegularizer/strided_slice³
!dense_64/ActivityRegularizer/CastCast3dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_64/ActivityRegularizer/CastÖ
$dense_64/ActivityRegularizer/truedivRealDiv5dense_64/ActivityRegularizer/PartitionedCall:output:0%dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_64/ActivityRegularizer/truediv¸
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_64_16615905*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulÔ
IdentityIdentity)dense_64/StatefulPartitionedCall:output:0!^dense_64/StatefulPartitionedCall2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_64/ActivityRegularizer/truediv:z:0!^dense_64/StatefulPartitionedCall2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_33
Á
¥
0__inference_sequential_65_layer_call_fn_16616528
dense_65_input
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_65_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_65_layer_call_and_return_conditional_losses_166159632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_65_input


+__inference_dense_65_layer_call_fn_16616675

inputs
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_166159502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
î"

K__inference_sequential_64_layer_call_and_return_conditional_losses_16615860

inputs#
dense_64_16615839:^ 
dense_64_16615841: 
identity

identity_1¢ dense_64/StatefulPartitionedCall¢1dense_64/kernel/Regularizer/Square/ReadVariableOp
 dense_64/StatefulPartitionedCallStatefulPartitionedCallinputsdense_64_16615839dense_64_16615841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_64_layer_call_and_return_conditional_losses_166157722"
 dense_64/StatefulPartitionedCallü
,dense_64/ActivityRegularizer/PartitionedCallPartitionedCall)dense_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_dense_64_activity_regularizer_166157482.
,dense_64/ActivityRegularizer/PartitionedCall¡
"dense_64/ActivityRegularizer/ShapeShape)dense_64/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_64/ActivityRegularizer/Shape®
0dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_64/ActivityRegularizer/strided_slice/stack²
2dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_1²
2dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_2
*dense_64/ActivityRegularizer/strided_sliceStridedSlice+dense_64/ActivityRegularizer/Shape:output:09dense_64/ActivityRegularizer/strided_slice/stack:output:0;dense_64/ActivityRegularizer/strided_slice/stack_1:output:0;dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_64/ActivityRegularizer/strided_slice³
!dense_64/ActivityRegularizer/CastCast3dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_64/ActivityRegularizer/CastÖ
$dense_64/ActivityRegularizer/truedivRealDiv5dense_64/ActivityRegularizer/PartitionedCall:output:0%dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_64/ActivityRegularizer/truediv¸
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_64_16615839*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulÔ
IdentityIdentity)dense_64/StatefulPartitionedCall:output:0!^dense_64/StatefulPartitionedCall2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_64/ActivityRegularizer/truediv:z:0!^dense_64/StatefulPartitionedCall2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
¨
Ç
J__inference_dense_64_layer_call_and_return_all_conditional_losses_16616649

inputs
unknown:^ 
	unknown_0: 
identity

identity_1¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_64_layer_call_and_return_conditional_losses_166157722
StatefulPartitionedCall¹
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_dense_64_activity_regularizer_166157482
PartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
Û
æ
$__inference__traced_restore_16616777
file_prefix2
 assignvariableop_dense_64_kernel:^ .
 assignvariableop_1_dense_64_bias: 4
"assignvariableop_2_dense_65_kernel: ^.
 assignvariableop_3_dense_65_bias:^

identity_5¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3ï
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*û
valueñBîB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesÄ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_64_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_64_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_65_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_65_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4¬

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
î"

K__inference_sequential_64_layer_call_and_return_conditional_losses_16615794

inputs#
dense_64_16615773:^ 
dense_64_16615775: 
identity

identity_1¢ dense_64/StatefulPartitionedCall¢1dense_64/kernel/Regularizer/Square/ReadVariableOp
 dense_64/StatefulPartitionedCallStatefulPartitionedCallinputsdense_64_16615773dense_64_16615775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_64_layer_call_and_return_conditional_losses_166157722"
 dense_64/StatefulPartitionedCallü
,dense_64/ActivityRegularizer/PartitionedCallPartitionedCall)dense_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_dense_64_activity_regularizer_166157482.
,dense_64/ActivityRegularizer/PartitionedCall¡
"dense_64/ActivityRegularizer/ShapeShape)dense_64/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_64/ActivityRegularizer/Shape®
0dense_64/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_64/ActivityRegularizer/strided_slice/stack²
2dense_64/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_1²
2dense_64/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_64/ActivityRegularizer/strided_slice/stack_2
*dense_64/ActivityRegularizer/strided_sliceStridedSlice+dense_64/ActivityRegularizer/Shape:output:09dense_64/ActivityRegularizer/strided_slice/stack:output:0;dense_64/ActivityRegularizer/strided_slice/stack_1:output:0;dense_64/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_64/ActivityRegularizer/strided_slice³
!dense_64/ActivityRegularizer/CastCast3dense_64/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_64/ActivityRegularizer/CastÖ
$dense_64/ActivityRegularizer/truedivRealDiv5dense_64/ActivityRegularizer/PartitionedCall:output:0%dense_64/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_64/ActivityRegularizer/truediv¸
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_64_16615773*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulÔ
IdentityIdentity)dense_64/StatefulPartitionedCall:output:0!^dense_64/StatefulPartitionedCall2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_64/ActivityRegularizer/truediv:z:0!^dense_64/StatefulPartitionedCall2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
¬

0__inference_sequential_64_layer_call_fn_16616421

inputs
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_64_layer_call_and_return_conditional_losses_166158602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
¯
Ô
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616589

inputs9
'dense_65_matmul_readvariableop_resource: ^6
(dense_65_biasadd_readvariableop_resource:^
identity¢dense_65/BiasAdd/ReadVariableOp¢dense_65/MatMul/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¨
dense_65/MatMul/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_65/MatMul/ReadVariableOp
dense_65/MatMulMatMulinputs&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/MatMul§
dense_65/BiasAdd/ReadVariableOpReadVariableOp(dense_65_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_65/BiasAdd/ReadVariableOp¥
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/BiasAdd|
dense_65/SigmoidSigmoiddense_65/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_65/SigmoidÎ
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_65_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulß
IdentityIdentitydense_65/Sigmoid:y:0 ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


K__inference_sequential_65_layer_call_and_return_conditional_losses_16615963

inputs#
dense_65_16615951: ^
dense_65_16615953:^
identity¢ dense_65/StatefulPartitionedCall¢1dense_65/kernel/Regularizer/Square/ReadVariableOp
 dense_65/StatefulPartitionedCallStatefulPartitionedCallinputsdense_65_16615951dense_65_16615953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_166159502"
 dense_65/StatefulPartitionedCall¸
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_65_16615951*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulÔ
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0!^dense_65/StatefulPartitionedCall2^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ò$
Î
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616140
x(
sequential_64_16616115:^ $
sequential_64_16616117: (
sequential_65_16616121: ^$
sequential_65_16616123:^
identity

identity_1¢1dense_64/kernel/Regularizer/Square/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¢%sequential_64/StatefulPartitionedCall¢%sequential_65/StatefulPartitionedCall±
%sequential_64/StatefulPartitionedCallStatefulPartitionedCallxsequential_64_16616115sequential_64_16616117*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_64_layer_call_and_return_conditional_losses_166158602'
%sequential_64/StatefulPartitionedCallÛ
%sequential_65/StatefulPartitionedCallStatefulPartitionedCall.sequential_64/StatefulPartitionedCall:output:0sequential_65_16616121sequential_65_16616123*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_65_layer_call_and_return_conditional_losses_166160062'
%sequential_65/StatefulPartitionedCall½
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_64_16616115*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mul½
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_65_16616121*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulº
IdentityIdentity.sequential_65/StatefulPartitionedCall:output:02^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp&^sequential_64/StatefulPartitionedCall&^sequential_65/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_64/StatefulPartitionedCall:output:12^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp&^sequential_64/StatefulPartitionedCall&^sequential_65/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_64/StatefulPartitionedCall%sequential_64/StatefulPartitionedCall2N
%sequential_65/StatefulPartitionedCall%sequential_65/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
ä
³
__inference_loss_fn_0_16616660L
:dense_64_kernel_regularizer_square_readvariableop_resource:^ 
identity¢1dense_64/kernel/Regularizer/Square/ReadVariableOpá
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_64_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mul
IdentityIdentity#dense_64/kernel/Regularizer/mul:z:02^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp
­
«
F__inference_dense_65_layer_call_and_return_conditional_losses_16616692

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2	
SigmoidÅ
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ò$
Î
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616084
x(
sequential_64_16616059:^ $
sequential_64_16616061: (
sequential_65_16616065: ^$
sequential_65_16616067:^
identity

identity_1¢1dense_64/kernel/Regularizer/Square/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¢%sequential_64/StatefulPartitionedCall¢%sequential_65/StatefulPartitionedCall±
%sequential_64/StatefulPartitionedCallStatefulPartitionedCallxsequential_64_16616059sequential_64_16616061*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_64_layer_call_and_return_conditional_losses_166157942'
%sequential_64/StatefulPartitionedCallÛ
%sequential_65/StatefulPartitionedCallStatefulPartitionedCall.sequential_64/StatefulPartitionedCall:output:0sequential_65_16616065sequential_65_16616067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_65_layer_call_and_return_conditional_losses_166159632'
%sequential_65/StatefulPartitionedCall½
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_64_16616059*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mul½
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_65_16616065*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulº
IdentityIdentity.sequential_65/StatefulPartitionedCall:output:02^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp&^sequential_64/StatefulPartitionedCall&^sequential_65/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_64/StatefulPartitionedCall:output:12^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp&^sequential_64/StatefulPartitionedCall&^sequential_65/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_64/StatefulPartitionedCall%sequential_64/StatefulPartitionedCall2N
%sequential_65/StatefulPartitionedCall%sequential_65/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
%
Ô
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616194
input_1(
sequential_64_16616169:^ $
sequential_64_16616171: (
sequential_65_16616175: ^$
sequential_65_16616177:^
identity

identity_1¢1dense_64/kernel/Regularizer/Square/ReadVariableOp¢1dense_65/kernel/Regularizer/Square/ReadVariableOp¢%sequential_64/StatefulPartitionedCall¢%sequential_65/StatefulPartitionedCall·
%sequential_64/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_64_16616169sequential_64_16616171*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_64_layer_call_and_return_conditional_losses_166157942'
%sequential_64/StatefulPartitionedCallÛ
%sequential_65/StatefulPartitionedCallStatefulPartitionedCall.sequential_64/StatefulPartitionedCall:output:0sequential_65_16616175sequential_65_16616177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_65_layer_call_and_return_conditional_losses_166159632'
%sequential_65/StatefulPartitionedCall½
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_64_16616169*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mul½
1dense_65/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_65_16616175*
_output_shapes

: ^*
dtype023
1dense_65/kernel/Regularizer/Square/ReadVariableOp¶
"dense_65/kernel/Regularizer/SquareSquare9dense_65/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_65/kernel/Regularizer/Square
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_65/kernel/Regularizer/Const¾
dense_65/kernel/Regularizer/SumSum&dense_65/kernel/Regularizer/Square:y:0*dense_65/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/Sum
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_65/kernel/Regularizer/mul/xÀ
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_65/kernel/Regularizer/mulº
IdentityIdentity.sequential_65/StatefulPartitionedCall:output:02^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp&^sequential_64/StatefulPartitionedCall&^sequential_65/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_64/StatefulPartitionedCall:output:12^dense_64/kernel/Regularizer/Square/ReadVariableOp2^dense_65/kernel/Regularizer/Square/ReadVariableOp&^sequential_64/StatefulPartitionedCall&^sequential_65/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp2f
1dense_65/kernel/Regularizer/Square/ReadVariableOp1dense_65/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_64/StatefulPartitionedCall%sequential_64/StatefulPartitionedCall2N
%sequential_65/StatefulPartitionedCall%sequential_65/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
²

0__inference_sequential_64_layer_call_fn_16615802
input_33
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_33unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_64_layer_call_and_return_conditional_losses_166157942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_33
­
«
F__inference_dense_64_layer_call_and_return_conditional_losses_16616720

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_64/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
SigmoidÅ
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs

Õ
1__inference_autoencoder_32_layer_call_fn_16616166
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_166161402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1

Õ
1__inference_autoencoder_32_layer_call_fn_16616096
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_166160842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
­
«
F__inference_dense_64_layer_call_and_return_conditional_losses_16615772

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_64/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
SigmoidÅ
1dense_64/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_64/kernel/Regularizer/Square/ReadVariableOp¶
"dense_64/kernel/Regularizer/SquareSquare9dense_64/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_64/kernel/Regularizer/Square
!dense_64/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_64/kernel/Regularizer/Const¾
dense_64/kernel/Regularizer/SumSum&dense_64/kernel/Regularizer/Square:y:0*dense_64/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/Sum
!dense_64/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_64/kernel/Regularizer/mul/xÀ
dense_64/kernel/Regularizer/mulMul*dense_64/kernel/Regularizer/mul/x:output:0(dense_64/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_64/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_64/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_64/kernel/Regularizer/Square/ReadVariableOp1dense_64/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs


+__inference_dense_64_layer_call_fn_16616638

inputs
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_64_layer_call_and_return_conditional_losses_166157722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
Ç
ª
!__inference__traced_save_16616755
file_prefix.
*savev2_dense_64_kernel_read_readvariableop,
(savev2_dense_64_bias_read_readvariableop.
*savev2_dense_65_kernel_read_readvariableop,
(savev2_dense_65_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameé
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*û
valueñBîB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesê
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_64_kernel_read_readvariableop(savev2_dense_64_bias_read_readvariableop*savev2_dense_65_kernel_read_readvariableop(savev2_dense_65_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*7
_input_shapes&
$: :^ : : ^:^: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:^ : 

_output_shapes
: :$ 

_output_shapes

: ^: 

_output_shapes
:^:

_output_shapes
: 
Á
¥
0__inference_sequential_65_layer_call_fn_16616555
dense_65_input
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_65_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_65_layer_call_and_return_conditional_losses_166160062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_65_input
©

0__inference_sequential_65_layer_call_fn_16616546

inputs
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_65_layer_call_and_return_conditional_losses_166160062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ^<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ^tensorflow/serving/predict:û±

history
encoder
decoder
trainable_variables
	variables
regularization_losses
	keras_api

signatures
8__call__
9_default_save_signature
*:&call_and_return_all_conditional_losses"§
_tf_keras_model{"name": "autoencoder_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
 "
trackable_dict_wrapper
³
	layer_with_weights-0
	layer-0

trainable_variables
	variables
regularization_losses
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"ý
_tf_keras_sequentialÞ{"name": "sequential_64", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_64", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_33"}}, {"class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_33"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_64", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_33"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
¾
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_sequentialé{"name": "sequential_65", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_65", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_65_input"}}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_65_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_65", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_65_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables

layers
	variables
layer_metrics
regularization_losses
8__call__
9_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
À

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"

_tf_keras_layer
{"name": "dense_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_64", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
­
 layer_regularization_losses
!non_trainable_variables
"metrics

trainable_variables

#layers
	variables
$layer_metrics
regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
í	

kernel
bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
C__call__
*D&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"name": "dense_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_65", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
­
)layer_regularization_losses
*non_trainable_variables
+metrics
trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
!:^ 2dense_64/kernel
: 2dense_64/bias
!: ^2dense_65/kernel
:^2dense_65/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
Ê
.layer_regularization_losses
/metrics
0non_trainable_variables
trainable_variables

1layers
	variables
2layer_metrics
regularization_losses
@__call__
Factivity_regularizer_fn
*A&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
­
3layer_regularization_losses
4metrics
5non_trainable_variables
%trainable_variables

6layers
&	variables
7layer_metrics
'regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
2ý
1__inference_autoencoder_32_layer_call_fn_16616096
1__inference_autoencoder_32_layer_call_fn_16616263
1__inference_autoencoder_32_layer_call_fn_16616277
1__inference_autoencoder_32_layer_call_fn_16616166®
¥²¡
FullArgSpec$
args
jself
jX

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
á2Þ
#__inference__wrapped_model_16615719¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ^
ì2é
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616336
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616395
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616194
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616222®
¥²¡
FullArgSpec$
args
jself
jX

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_sequential_64_layer_call_fn_16615802
0__inference_sequential_64_layer_call_fn_16616411
0__inference_sequential_64_layer_call_fn_16616421
0__inference_sequential_64_layer_call_fn_16615878À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_sequential_64_layer_call_and_return_conditional_losses_16616467
K__inference_sequential_64_layer_call_and_return_conditional_losses_16616513
K__inference_sequential_64_layer_call_and_return_conditional_losses_16615902
K__inference_sequential_64_layer_call_and_return_conditional_losses_16615926À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_sequential_65_layer_call_fn_16616528
0__inference_sequential_65_layer_call_fn_16616537
0__inference_sequential_65_layer_call_fn_16616546
0__inference_sequential_65_layer_call_fn_16616555À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616572
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616589
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616606
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616623À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÍBÊ
&__inference_signature_wrapper_16616249input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_64_layer_call_fn_16616638¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_dense_64_layer_call_and_return_all_conditional_losses_16616649¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²
__inference_loss_fn_0_16616660
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
Õ2Ò
+__inference_dense_65_layer_call_fn_16616675¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_65_layer_call_and_return_conditional_losses_16616692¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²
__inference_loss_fn_1_16616703
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ì2é
2__inference_dense_64_activity_regularizer_16615748²
²
FullArgSpec!
args
jself
j
activation
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ð2í
F__inference_dense_64_layer_call_and_return_conditional_losses_16616720¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#__inference__wrapped_model_16615719m0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ^
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ^Á
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616194q4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ^

	
1/0 Á
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616222q4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ^

	
1/0 »
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616336k.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ^

	
1/0 »
L__inference_autoencoder_32_layer_call_and_return_conditional_losses_16616395k.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ^

	
1/0 
1__inference_autoencoder_32_layer_call_fn_16616096V4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p 
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_32_layer_call_fn_16616166V4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_32_layer_call_fn_16616263P.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p 
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_32_layer_call_fn_16616277P.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p
ª "ÿÿÿÿÿÿÿÿÿ^e
2__inference_dense_64_activity_regularizer_16615748/$¢!
¢


activation
ª " ¸
J__inference_dense_64_layer_call_and_return_all_conditional_losses_16616649j/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 ¦
F__inference_dense_64_layer_call_and_return_conditional_losses_16616720\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_dense_64_layer_call_fn_16616638O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "ÿÿÿÿÿÿÿÿÿ ¦
F__inference_dense_65_layer_call_and_return_conditional_losses_16616692\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 ~
+__inference_dense_65_layer_call_fn_16616675O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ^=
__inference_loss_fn_0_16616660¢

¢ 
ª " =
__inference_loss_fn_1_16616703¢

¢ 
ª " Ã
K__inference_sequential_64_layer_call_and_return_conditional_losses_16615902t9¢6
/¢,
"
input_33ÿÿÿÿÿÿÿÿÿ^
p 

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Ã
K__inference_sequential_64_layer_call_and_return_conditional_losses_16615926t9¢6
/¢,
"
input_33ÿÿÿÿÿÿÿÿÿ^
p

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Á
K__inference_sequential_64_layer_call_and_return_conditional_losses_16616467r7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p 

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Á
K__inference_sequential_64_layer_call_and_return_conditional_losses_16616513r7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 
0__inference_sequential_64_layer_call_fn_16615802Y9¢6
/¢,
"
input_33ÿÿÿÿÿÿÿÿÿ^
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_64_layer_call_fn_16615878Y9¢6
/¢,
"
input_33ÿÿÿÿÿÿÿÿÿ^
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_64_layer_call_fn_16616411W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_64_layer_call_fn_16616421W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p

 
ª "ÿÿÿÿÿÿÿÿÿ ³
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616572d7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 ³
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616589d7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 »
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616606l?¢<
5¢2
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 »
K__inference_sequential_65_layer_call_and_return_conditional_losses_16616623l?¢<
5¢2
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 
0__inference_sequential_65_layer_call_fn_16616528_?¢<
5¢2
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_65_layer_call_fn_16616537W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_65_layer_call_fn_16616546W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_65_layer_call_fn_16616555_?¢<
5¢2
(%
dense_65_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ^¢
&__inference_signature_wrapper_16616249x;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ^"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ^