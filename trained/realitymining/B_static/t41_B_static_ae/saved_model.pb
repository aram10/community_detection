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
dense_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ * 
shared_namedense_80/kernel
s
#dense_80/kernel/Read/ReadVariableOpReadVariableOpdense_80/kernel*
_output_shapes

:^ *
dtype0
r
dense_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_80/bias
k
!dense_80/bias/Read/ReadVariableOpReadVariableOpdense_80/bias*
_output_shapes
: *
dtype0
z
dense_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^* 
shared_namedense_81/kernel
s
#dense_81/kernel/Read/ReadVariableOpReadVariableOpdense_81/kernel*
_output_shapes

: ^*
dtype0
r
dense_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_81/bias
k
!dense_81/bias/Read/ReadVariableOpReadVariableOpdense_81/bias*
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
VARIABLE_VALUEdense_80/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_80/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_81/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_81/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_80/kerneldense_80/biasdense_81/kerneldense_81/bias*
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
&__inference_signature_wrapper_16626257
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
±
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_80/kernel/Read/ReadVariableOp!dense_80/bias/Read/ReadVariableOp#dense_81/kernel/Read/ReadVariableOp!dense_81/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_16626763
Ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_80/kerneldense_80/biasdense_81/kerneldense_81/bias*
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
$__inference__traced_restore_16626785¥ô
%
Ô
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626202
input_1(
sequential_80_16626177:^ $
sequential_80_16626179: (
sequential_81_16626183: ^$
sequential_81_16626185:^
identity

identity_1¢1dense_80/kernel/Regularizer/Square/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¢%sequential_80/StatefulPartitionedCall¢%sequential_81/StatefulPartitionedCall·
%sequential_80/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_80_16626177sequential_80_16626179*
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
K__inference_sequential_80_layer_call_and_return_conditional_losses_166258022'
%sequential_80/StatefulPartitionedCallÛ
%sequential_81/StatefulPartitionedCallStatefulPartitionedCall.sequential_80/StatefulPartitionedCall:output:0sequential_81_16626183sequential_81_16626185*
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_166259712'
%sequential_81/StatefulPartitionedCall½
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_80_16626177*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mul½
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_81_16626183*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulº
IdentityIdentity.sequential_81/StatefulPartitionedCall:output:02^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp&^sequential_80/StatefulPartitionedCall&^sequential_81/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_80/StatefulPartitionedCall:output:12^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp&^sequential_80/StatefulPartitionedCall&^sequential_81/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_80/StatefulPartitionedCall%sequential_80/StatefulPartitionedCall2N
%sequential_81/StatefulPartitionedCall%sequential_81/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
²A
ä
K__inference_sequential_80_layer_call_and_return_conditional_losses_16626475

inputs9
'dense_80_matmul_readvariableop_resource:^ 6
(dense_80_biasadd_readvariableop_resource: 
identity

identity_1¢dense_80/BiasAdd/ReadVariableOp¢dense_80/MatMul/ReadVariableOp¢1dense_80/kernel/Regularizer/Square/ReadVariableOp¨
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02 
dense_80/MatMul/ReadVariableOp
dense_80/MatMulMatMulinputs&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_80/MatMul§
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_80/BiasAdd/ReadVariableOp¥
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_80/BiasAdd|
dense_80/SigmoidSigmoiddense_80/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_80/Sigmoid¬
3dense_80/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_80/ActivityRegularizer/Mean/reduction_indicesÇ
!dense_80/ActivityRegularizer/MeanMeandense_80/Sigmoid:y:0<dense_80/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2#
!dense_80/ActivityRegularizer/Mean
&dense_80/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2(
&dense_80/ActivityRegularizer/Maximum/yÙ
$dense_80/ActivityRegularizer/MaximumMaximum*dense_80/ActivityRegularizer/Mean:output:0/dense_80/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2&
$dense_80/ActivityRegularizer/Maximum
&dense_80/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&dense_80/ActivityRegularizer/truediv/x×
$dense_80/ActivityRegularizer/truedivRealDiv/dense_80/ActivityRegularizer/truediv/x:output:0(dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2&
$dense_80/ActivityRegularizer/truediv
 dense_80/ActivityRegularizer/LogLog(dense_80/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/Log
"dense_80/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"dense_80/ActivityRegularizer/mul/xÃ
 dense_80/ActivityRegularizer/mulMul+dense_80/ActivityRegularizer/mul/x:output:0$dense_80/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/mul
"dense_80/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dense_80/ActivityRegularizer/sub/xÇ
 dense_80/ActivityRegularizer/subSub+dense_80/ActivityRegularizer/sub/x:output:0(dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/sub
(dense_80/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2*
(dense_80/ActivityRegularizer/truediv_1/xÙ
&dense_80/ActivityRegularizer/truediv_1RealDiv1dense_80/ActivityRegularizer/truediv_1/x:output:0$dense_80/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2(
&dense_80/ActivityRegularizer/truediv_1 
"dense_80/ActivityRegularizer/Log_1Log*dense_80/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2$
"dense_80/ActivityRegularizer/Log_1
$dense_80/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2&
$dense_80/ActivityRegularizer/mul_1/xË
"dense_80/ActivityRegularizer/mul_1Mul-dense_80/ActivityRegularizer/mul_1/x:output:0&dense_80/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2$
"dense_80/ActivityRegularizer/mul_1À
 dense_80/ActivityRegularizer/addAddV2$dense_80/ActivityRegularizer/mul:z:0&dense_80/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/add
"dense_80/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_80/ActivityRegularizer/Const¿
 dense_80/ActivityRegularizer/SumSum$dense_80/ActivityRegularizer/add:z:0+dense_80/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/Sum
$dense_80/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dense_80/ActivityRegularizer/mul_2/xÊ
"dense_80/ActivityRegularizer/mul_2Mul-dense_80/ActivityRegularizer/mul_2/x:output:0)dense_80/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_80/ActivityRegularizer/mul_2
"dense_80/ActivityRegularizer/ShapeShapedense_80/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_80/ActivityRegularizer/Shape®
0dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_80/ActivityRegularizer/strided_slice/stack²
2dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_1²
2dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_2
*dense_80/ActivityRegularizer/strided_sliceStridedSlice+dense_80/ActivityRegularizer/Shape:output:09dense_80/ActivityRegularizer/strided_slice/stack:output:0;dense_80/ActivityRegularizer/strided_slice/stack_1:output:0;dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_80/ActivityRegularizer/strided_slice³
!dense_80/ActivityRegularizer/CastCast3dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_80/ActivityRegularizer/CastË
&dense_80/ActivityRegularizer/truediv_2RealDiv&dense_80/ActivityRegularizer/mul_2:z:0%dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_80/ActivityRegularizer/truediv_2Î
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulß
IdentityIdentitydense_80/Sigmoid:y:0 ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityè

Identity_1Identity*dense_80/ActivityRegularizer/truediv_2:z:0 ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs


K__inference_sequential_81_layer_call_and_return_conditional_losses_16625971

inputs#
dense_81_16625959: ^
dense_81_16625961:^
identity¢ dense_81/StatefulPartitionedCall¢1dense_81/kernel/Regularizer/Square/ReadVariableOp
 dense_81/StatefulPartitionedCallStatefulPartitionedCallinputsdense_81_16625959dense_81_16625961*
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
F__inference_dense_81_layer_call_and_return_conditional_losses_166259582"
 dense_81/StatefulPartitionedCall¸
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_81_16625959*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulÔ
IdentityIdentity)dense_81/StatefulPartitionedCall:output:0!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó
Ï
1__inference_autoencoder_40_layer_call_fn_16626285
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
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_166261482
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
©

0__inference_sequential_81_layer_call_fn_16626545

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
K__inference_sequential_81_layer_call_and_return_conditional_losses_166259712
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
¯
Ô
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626597

inputs9
'dense_81_matmul_readvariableop_resource: ^6
(dense_81_biasadd_readvariableop_resource:^
identity¢dense_81/BiasAdd/ReadVariableOp¢dense_81/MatMul/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¨
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_81/MatMul/ReadVariableOp
dense_81/MatMulMatMulinputs&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/MatMul§
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_81/BiasAdd/ReadVariableOp¥
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/BiasAdd|
dense_81/SigmoidSigmoiddense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/SigmoidÎ
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulß
IdentityIdentitydense_81/Sigmoid:y:0 ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬

0__inference_sequential_80_layer_call_fn_16626429

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
K__inference_sequential_80_layer_call_and_return_conditional_losses_166258682
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
ò$
Î
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626092
x(
sequential_80_16626067:^ $
sequential_80_16626069: (
sequential_81_16626073: ^$
sequential_81_16626075:^
identity

identity_1¢1dense_80/kernel/Regularizer/Square/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¢%sequential_80/StatefulPartitionedCall¢%sequential_81/StatefulPartitionedCall±
%sequential_80/StatefulPartitionedCallStatefulPartitionedCallxsequential_80_16626067sequential_80_16626069*
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
K__inference_sequential_80_layer_call_and_return_conditional_losses_166258022'
%sequential_80/StatefulPartitionedCallÛ
%sequential_81/StatefulPartitionedCallStatefulPartitionedCall.sequential_80/StatefulPartitionedCall:output:0sequential_81_16626073sequential_81_16626075*
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_166259712'
%sequential_81/StatefulPartitionedCall½
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_80_16626067*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mul½
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_81_16626073*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulº
IdentityIdentity.sequential_81/StatefulPartitionedCall:output:02^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp&^sequential_80/StatefulPartitionedCall&^sequential_81/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_80/StatefulPartitionedCall:output:12^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp&^sequential_80/StatefulPartitionedCall&^sequential_81/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_80/StatefulPartitionedCall%sequential_80/StatefulPartitionedCall2N
%sequential_81/StatefulPartitionedCall%sequential_81/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
¯
Ô
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626580

inputs9
'dense_81_matmul_readvariableop_resource: ^6
(dense_81_biasadd_readvariableop_resource:^
identity¢dense_81/BiasAdd/ReadVariableOp¢dense_81/MatMul/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¨
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_81/MatMul/ReadVariableOp
dense_81/MatMulMatMulinputs&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/MatMul§
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_81/BiasAdd/ReadVariableOp¥
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/BiasAdd|
dense_81/SigmoidSigmoiddense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/SigmoidÎ
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulß
IdentityIdentitydense_81/Sigmoid:y:0 ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
î"

K__inference_sequential_80_layer_call_and_return_conditional_losses_16625802

inputs#
dense_80_16625781:^ 
dense_80_16625783: 
identity

identity_1¢ dense_80/StatefulPartitionedCall¢1dense_80/kernel/Regularizer/Square/ReadVariableOp
 dense_80/StatefulPartitionedCallStatefulPartitionedCallinputsdense_80_16625781dense_80_16625783*
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
F__inference_dense_80_layer_call_and_return_conditional_losses_166257802"
 dense_80/StatefulPartitionedCallü
,dense_80/ActivityRegularizer/PartitionedCallPartitionedCall)dense_80/StatefulPartitionedCall:output:0*
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
2__inference_dense_80_activity_regularizer_166257562.
,dense_80/ActivityRegularizer/PartitionedCall¡
"dense_80/ActivityRegularizer/ShapeShape)dense_80/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_80/ActivityRegularizer/Shape®
0dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_80/ActivityRegularizer/strided_slice/stack²
2dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_1²
2dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_2
*dense_80/ActivityRegularizer/strided_sliceStridedSlice+dense_80/ActivityRegularizer/Shape:output:09dense_80/ActivityRegularizer/strided_slice/stack:output:0;dense_80/ActivityRegularizer/strided_slice/stack_1:output:0;dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_80/ActivityRegularizer/strided_slice³
!dense_80/ActivityRegularizer/CastCast3dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_80/ActivityRegularizer/CastÖ
$dense_80/ActivityRegularizer/truedivRealDiv5dense_80/ActivityRegularizer/PartitionedCall:output:0%dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_80/ActivityRegularizer/truediv¸
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_80_16625781*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulÔ
IdentityIdentity)dense_80/StatefulPartitionedCall:output:0!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_80/ActivityRegularizer/truediv:z:0!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
Á
¥
0__inference_sequential_81_layer_call_fn_16626563
dense_81_input
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_81_inputunknown	unknown_0*
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_166260142
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
_user_specified_namedense_81_input
©

0__inference_sequential_81_layer_call_fn_16626554

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
K__inference_sequential_81_layer_call_and_return_conditional_losses_166260142
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
F__inference_dense_80_layer_call_and_return_conditional_losses_16626728

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_80/kernel/Regularizer/Square/ReadVariableOp
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
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
ä
³
__inference_loss_fn_1_16626711L
:dense_81_kernel_regularizer_square_readvariableop_resource: ^
identity¢1dense_81/kernel/Regularizer/Square/ReadVariableOpá
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_81_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mul
IdentityIdentity#dense_81/kernel/Regularizer/mul:z:02^dense_81/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp
Á
¥
0__inference_sequential_81_layer_call_fn_16626536
dense_81_input
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_81_inputunknown	unknown_0*
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_166259712
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
_user_specified_namedense_81_input


+__inference_dense_80_layer_call_fn_16626646

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
F__inference_dense_80_layer_call_and_return_conditional_losses_166257802
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

Õ
1__inference_autoencoder_40_layer_call_fn_16626104
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
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_166260922
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
F__inference_dense_81_layer_call_and_return_conditional_losses_16625958

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp
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
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ò$
Î
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626148
x(
sequential_80_16626123:^ $
sequential_80_16626125: (
sequential_81_16626129: ^$
sequential_81_16626131:^
identity

identity_1¢1dense_80/kernel/Regularizer/Square/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¢%sequential_80/StatefulPartitionedCall¢%sequential_81/StatefulPartitionedCall±
%sequential_80/StatefulPartitionedCallStatefulPartitionedCallxsequential_80_16626123sequential_80_16626125*
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
K__inference_sequential_80_layer_call_and_return_conditional_losses_166258682'
%sequential_80/StatefulPartitionedCallÛ
%sequential_81/StatefulPartitionedCallStatefulPartitionedCall.sequential_80/StatefulPartitionedCall:output:0sequential_81_16626129sequential_81_16626131*
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_166260142'
%sequential_81/StatefulPartitionedCall½
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_80_16626123*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mul½
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_81_16626129*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulº
IdentityIdentity.sequential_81/StatefulPartitionedCall:output:02^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp&^sequential_80/StatefulPartitionedCall&^sequential_81/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_80/StatefulPartitionedCall:output:12^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp&^sequential_80/StatefulPartitionedCall&^sequential_81/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_80/StatefulPartitionedCall%sequential_80/StatefulPartitionedCall2N
%sequential_81/StatefulPartitionedCall%sequential_81/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
î"

K__inference_sequential_80_layer_call_and_return_conditional_losses_16625868

inputs#
dense_80_16625847:^ 
dense_80_16625849: 
identity

identity_1¢ dense_80/StatefulPartitionedCall¢1dense_80/kernel/Regularizer/Square/ReadVariableOp
 dense_80/StatefulPartitionedCallStatefulPartitionedCallinputsdense_80_16625847dense_80_16625849*
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
F__inference_dense_80_layer_call_and_return_conditional_losses_166257802"
 dense_80/StatefulPartitionedCallü
,dense_80/ActivityRegularizer/PartitionedCallPartitionedCall)dense_80/StatefulPartitionedCall:output:0*
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
2__inference_dense_80_activity_regularizer_166257562.
,dense_80/ActivityRegularizer/PartitionedCall¡
"dense_80/ActivityRegularizer/ShapeShape)dense_80/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_80/ActivityRegularizer/Shape®
0dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_80/ActivityRegularizer/strided_slice/stack²
2dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_1²
2dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_2
*dense_80/ActivityRegularizer/strided_sliceStridedSlice+dense_80/ActivityRegularizer/Shape:output:09dense_80/ActivityRegularizer/strided_slice/stack:output:0;dense_80/ActivityRegularizer/strided_slice/stack_1:output:0;dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_80/ActivityRegularizer/strided_slice³
!dense_80/ActivityRegularizer/CastCast3dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_80/ActivityRegularizer/CastÖ
$dense_80/ActivityRegularizer/truedivRealDiv5dense_80/ActivityRegularizer/PartitionedCall:output:0%dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_80/ActivityRegularizer/truediv¸
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_80_16625847*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulÔ
IdentityIdentity)dense_80/StatefulPartitionedCall:output:0!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_80/ActivityRegularizer/truediv:z:0!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
ô"

K__inference_sequential_80_layer_call_and_return_conditional_losses_16625910
input_41#
dense_80_16625889:^ 
dense_80_16625891: 
identity

identity_1¢ dense_80/StatefulPartitionedCall¢1dense_80/kernel/Regularizer/Square/ReadVariableOp
 dense_80/StatefulPartitionedCallStatefulPartitionedCallinput_41dense_80_16625889dense_80_16625891*
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
F__inference_dense_80_layer_call_and_return_conditional_losses_166257802"
 dense_80/StatefulPartitionedCallü
,dense_80/ActivityRegularizer/PartitionedCallPartitionedCall)dense_80/StatefulPartitionedCall:output:0*
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
2__inference_dense_80_activity_regularizer_166257562.
,dense_80/ActivityRegularizer/PartitionedCall¡
"dense_80/ActivityRegularizer/ShapeShape)dense_80/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_80/ActivityRegularizer/Shape®
0dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_80/ActivityRegularizer/strided_slice/stack²
2dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_1²
2dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_2
*dense_80/ActivityRegularizer/strided_sliceStridedSlice+dense_80/ActivityRegularizer/Shape:output:09dense_80/ActivityRegularizer/strided_slice/stack:output:0;dense_80/ActivityRegularizer/strided_slice/stack_1:output:0;dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_80/ActivityRegularizer/strided_slice³
!dense_80/ActivityRegularizer/CastCast3dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_80/ActivityRegularizer/CastÖ
$dense_80/ActivityRegularizer/truedivRealDiv5dense_80/ActivityRegularizer/PartitionedCall:output:0%dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_80/ActivityRegularizer/truediv¸
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_80_16625889*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulÔ
IdentityIdentity)dense_80/StatefulPartitionedCall:output:0!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_80/ActivityRegularizer/truediv:z:0!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_41
ô"

K__inference_sequential_80_layer_call_and_return_conditional_losses_16625934
input_41#
dense_80_16625913:^ 
dense_80_16625915: 
identity

identity_1¢ dense_80/StatefulPartitionedCall¢1dense_80/kernel/Regularizer/Square/ReadVariableOp
 dense_80/StatefulPartitionedCallStatefulPartitionedCallinput_41dense_80_16625913dense_80_16625915*
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
F__inference_dense_80_layer_call_and_return_conditional_losses_166257802"
 dense_80/StatefulPartitionedCallü
,dense_80/ActivityRegularizer/PartitionedCallPartitionedCall)dense_80/StatefulPartitionedCall:output:0*
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
2__inference_dense_80_activity_regularizer_166257562.
,dense_80/ActivityRegularizer/PartitionedCall¡
"dense_80/ActivityRegularizer/ShapeShape)dense_80/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_80/ActivityRegularizer/Shape®
0dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_80/ActivityRegularizer/strided_slice/stack²
2dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_1²
2dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_2
*dense_80/ActivityRegularizer/strided_sliceStridedSlice+dense_80/ActivityRegularizer/Shape:output:09dense_80/ActivityRegularizer/strided_slice/stack:output:0;dense_80/ActivityRegularizer/strided_slice/stack_1:output:0;dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_80/ActivityRegularizer/strided_slice³
!dense_80/ActivityRegularizer/CastCast3dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_80/ActivityRegularizer/CastÖ
$dense_80/ActivityRegularizer/truedivRealDiv5dense_80/ActivityRegularizer/PartitionedCall:output:0%dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_80/ActivityRegularizer/truediv¸
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_80_16625913*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulÔ
IdentityIdentity)dense_80/StatefulPartitionedCall:output:0!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_80/ActivityRegularizer/truediv:z:0!^dense_80/StatefulPartitionedCall2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_41
Ç
ª
!__inference__traced_save_16626763
file_prefix.
*savev2_dense_80_kernel_read_readvariableop,
(savev2_dense_80_bias_read_readvariableop.
*savev2_dense_81_kernel_read_readvariableop,
(savev2_dense_81_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_80_kernel_read_readvariableop(savev2_dense_80_bias_read_readvariableop*savev2_dense_81_kernel_read_readvariableop(savev2_dense_81_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
%
Ô
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626230
input_1(
sequential_80_16626205:^ $
sequential_80_16626207: (
sequential_81_16626211: ^$
sequential_81_16626213:^
identity

identity_1¢1dense_80/kernel/Regularizer/Square/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¢%sequential_80/StatefulPartitionedCall¢%sequential_81/StatefulPartitionedCall·
%sequential_80/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_80_16626205sequential_80_16626207*
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
K__inference_sequential_80_layer_call_and_return_conditional_losses_166258682'
%sequential_80/StatefulPartitionedCallÛ
%sequential_81/StatefulPartitionedCallStatefulPartitionedCall.sequential_80/StatefulPartitionedCall:output:0sequential_81_16626211sequential_81_16626213*
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_166260142'
%sequential_81/StatefulPartitionedCall½
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_80_16626205*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mul½
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_81_16626211*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulº
IdentityIdentity.sequential_81/StatefulPartitionedCall:output:02^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp&^sequential_80/StatefulPartitionedCall&^sequential_81/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_80/StatefulPartitionedCall:output:12^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp&^sequential_80/StatefulPartitionedCall&^sequential_81/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_80/StatefulPartitionedCall%sequential_80/StatefulPartitionedCall2N
%sequential_81/StatefulPartitionedCall%sequential_81/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
Û
æ
$__inference__traced_restore_16626785
file_prefix2
 assignvariableop_dense_80_kernel:^ .
 assignvariableop_1_dense_80_bias: 4
"assignvariableop_2_dense_81_kernel: ^.
 assignvariableop_3_dense_81_bias:^

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
AssignVariableOpAssignVariableOp assignvariableop_dense_80_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_80_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_81_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_81_biasIdentity_3:output:0"/device:CPU:0*
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
Êe
º
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626403
xG
5sequential_80_dense_80_matmul_readvariableop_resource:^ D
6sequential_80_dense_80_biasadd_readvariableop_resource: G
5sequential_81_dense_81_matmul_readvariableop_resource: ^D
6sequential_81_dense_81_biasadd_readvariableop_resource:^
identity

identity_1¢1dense_80/kernel/Regularizer/Square/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¢-sequential_80/dense_80/BiasAdd/ReadVariableOp¢,sequential_80/dense_80/MatMul/ReadVariableOp¢-sequential_81/dense_81/BiasAdd/ReadVariableOp¢,sequential_81/dense_81/MatMul/ReadVariableOpÒ
,sequential_80/dense_80/MatMul/ReadVariableOpReadVariableOp5sequential_80_dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02.
,sequential_80/dense_80/MatMul/ReadVariableOp³
sequential_80/dense_80/MatMulMatMulx4sequential_80/dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_80/dense_80/MatMulÑ
-sequential_80/dense_80/BiasAdd/ReadVariableOpReadVariableOp6sequential_80_dense_80_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_80/dense_80/BiasAdd/ReadVariableOpÝ
sequential_80/dense_80/BiasAddBiasAdd'sequential_80/dense_80/MatMul:product:05sequential_80/dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_80/dense_80/BiasAdd¦
sequential_80/dense_80/SigmoidSigmoid'sequential_80/dense_80/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_80/dense_80/SigmoidÈ
Asequential_80/dense_80/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_80/dense_80/ActivityRegularizer/Mean/reduction_indicesÿ
/sequential_80/dense_80/ActivityRegularizer/MeanMean"sequential_80/dense_80/Sigmoid:y:0Jsequential_80/dense_80/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 21
/sequential_80/dense_80/ActivityRegularizer/Mean±
4sequential_80/dense_80/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.26
4sequential_80/dense_80/ActivityRegularizer/Maximum/y
2sequential_80/dense_80/ActivityRegularizer/MaximumMaximum8sequential_80/dense_80/ActivityRegularizer/Mean:output:0=sequential_80/dense_80/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 24
2sequential_80/dense_80/ActivityRegularizer/Maximum±
4sequential_80/dense_80/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<26
4sequential_80/dense_80/ActivityRegularizer/truediv/x
2sequential_80/dense_80/ActivityRegularizer/truedivRealDiv=sequential_80/dense_80/ActivityRegularizer/truediv/x:output:06sequential_80/dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 24
2sequential_80/dense_80/ActivityRegularizer/truedivÄ
.sequential_80/dense_80/ActivityRegularizer/LogLog6sequential_80/dense_80/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/Log©
0sequential_80/dense_80/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<22
0sequential_80/dense_80/ActivityRegularizer/mul/xû
.sequential_80/dense_80/ActivityRegularizer/mulMul9sequential_80/dense_80/ActivityRegularizer/mul/x:output:02sequential_80/dense_80/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/mul©
0sequential_80/dense_80/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_80/dense_80/ActivityRegularizer/sub/xÿ
.sequential_80/dense_80/ActivityRegularizer/subSub9sequential_80/dense_80/ActivityRegularizer/sub/x:output:06sequential_80/dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/subµ
6sequential_80/dense_80/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?28
6sequential_80/dense_80/ActivityRegularizer/truediv_1/x
4sequential_80/dense_80/ActivityRegularizer/truediv_1RealDiv?sequential_80/dense_80/ActivityRegularizer/truediv_1/x:output:02sequential_80/dense_80/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 26
4sequential_80/dense_80/ActivityRegularizer/truediv_1Ê
0sequential_80/dense_80/ActivityRegularizer/Log_1Log8sequential_80/dense_80/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 22
0sequential_80/dense_80/ActivityRegularizer/Log_1­
2sequential_80/dense_80/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?24
2sequential_80/dense_80/ActivityRegularizer/mul_1/x
0sequential_80/dense_80/ActivityRegularizer/mul_1Mul;sequential_80/dense_80/ActivityRegularizer/mul_1/x:output:04sequential_80/dense_80/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 22
0sequential_80/dense_80/ActivityRegularizer/mul_1ø
.sequential_80/dense_80/ActivityRegularizer/addAddV22sequential_80/dense_80/ActivityRegularizer/mul:z:04sequential_80/dense_80/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/add®
0sequential_80/dense_80/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_80/dense_80/ActivityRegularizer/Const÷
.sequential_80/dense_80/ActivityRegularizer/SumSum2sequential_80/dense_80/ActivityRegularizer/add:z:09sequential_80/dense_80/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/Sum­
2sequential_80/dense_80/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_80/dense_80/ActivityRegularizer/mul_2/x
0sequential_80/dense_80/ActivityRegularizer/mul_2Mul;sequential_80/dense_80/ActivityRegularizer/mul_2/x:output:07sequential_80/dense_80/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_80/dense_80/ActivityRegularizer/mul_2¶
0sequential_80/dense_80/ActivityRegularizer/ShapeShape"sequential_80/dense_80/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_80/dense_80/ActivityRegularizer/ShapeÊ
>sequential_80/dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_80/dense_80/ActivityRegularizer/strided_slice/stackÎ
@sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1Î
@sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2ä
8sequential_80/dense_80/ActivityRegularizer/strided_sliceStridedSlice9sequential_80/dense_80/ActivityRegularizer/Shape:output:0Gsequential_80/dense_80/ActivityRegularizer/strided_slice/stack:output:0Isequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_80/dense_80/ActivityRegularizer/strided_sliceÝ
/sequential_80/dense_80/ActivityRegularizer/CastCastAsequential_80/dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_80/dense_80/ActivityRegularizer/Cast
4sequential_80/dense_80/ActivityRegularizer/truediv_2RealDiv4sequential_80/dense_80/ActivityRegularizer/mul_2:z:03sequential_80/dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_80/dense_80/ActivityRegularizer/truediv_2Ò
,sequential_81/dense_81/MatMul/ReadVariableOpReadVariableOp5sequential_81_dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02.
,sequential_81/dense_81/MatMul/ReadVariableOpÔ
sequential_81/dense_81/MatMulMatMul"sequential_80/dense_80/Sigmoid:y:04sequential_81/dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
sequential_81/dense_81/MatMulÑ
-sequential_81/dense_81/BiasAdd/ReadVariableOpReadVariableOp6sequential_81_dense_81_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02/
-sequential_81/dense_81/BiasAdd/ReadVariableOpÝ
sequential_81/dense_81/BiasAddBiasAdd'sequential_81/dense_81/MatMul:product:05sequential_81/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_81/dense_81/BiasAdd¦
sequential_81/dense_81/SigmoidSigmoid'sequential_81/dense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_81/dense_81/SigmoidÜ
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_80_dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulÜ
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_81_dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mul
IdentityIdentity"sequential_81/dense_81/Sigmoid:y:02^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp.^sequential_80/dense_80/BiasAdd/ReadVariableOp-^sequential_80/dense_80/MatMul/ReadVariableOp.^sequential_81/dense_81/BiasAdd/ReadVariableOp-^sequential_81/dense_81/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity¥

Identity_1Identity8sequential_80/dense_80/ActivityRegularizer/truediv_2:z:02^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp.^sequential_80/dense_80/BiasAdd/ReadVariableOp-^sequential_80/dense_80/MatMul/ReadVariableOp.^sequential_81/dense_81/BiasAdd/ReadVariableOp-^sequential_81/dense_81/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_80/dense_80/BiasAdd/ReadVariableOp-sequential_80/dense_80/BiasAdd/ReadVariableOp2\
,sequential_80/dense_80/MatMul/ReadVariableOp,sequential_80/dense_80/MatMul/ReadVariableOp2^
-sequential_81/dense_81/BiasAdd/ReadVariableOp-sequential_81/dense_81/BiasAdd/ReadVariableOp2\
,sequential_81/dense_81/MatMul/ReadVariableOp,sequential_81/dense_81/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
Êe
º
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626344
xG
5sequential_80_dense_80_matmul_readvariableop_resource:^ D
6sequential_80_dense_80_biasadd_readvariableop_resource: G
5sequential_81_dense_81_matmul_readvariableop_resource: ^D
6sequential_81_dense_81_biasadd_readvariableop_resource:^
identity

identity_1¢1dense_80/kernel/Regularizer/Square/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¢-sequential_80/dense_80/BiasAdd/ReadVariableOp¢,sequential_80/dense_80/MatMul/ReadVariableOp¢-sequential_81/dense_81/BiasAdd/ReadVariableOp¢,sequential_81/dense_81/MatMul/ReadVariableOpÒ
,sequential_80/dense_80/MatMul/ReadVariableOpReadVariableOp5sequential_80_dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02.
,sequential_80/dense_80/MatMul/ReadVariableOp³
sequential_80/dense_80/MatMulMatMulx4sequential_80/dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_80/dense_80/MatMulÑ
-sequential_80/dense_80/BiasAdd/ReadVariableOpReadVariableOp6sequential_80_dense_80_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_80/dense_80/BiasAdd/ReadVariableOpÝ
sequential_80/dense_80/BiasAddBiasAdd'sequential_80/dense_80/MatMul:product:05sequential_80/dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_80/dense_80/BiasAdd¦
sequential_80/dense_80/SigmoidSigmoid'sequential_80/dense_80/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_80/dense_80/SigmoidÈ
Asequential_80/dense_80/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_80/dense_80/ActivityRegularizer/Mean/reduction_indicesÿ
/sequential_80/dense_80/ActivityRegularizer/MeanMean"sequential_80/dense_80/Sigmoid:y:0Jsequential_80/dense_80/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 21
/sequential_80/dense_80/ActivityRegularizer/Mean±
4sequential_80/dense_80/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.26
4sequential_80/dense_80/ActivityRegularizer/Maximum/y
2sequential_80/dense_80/ActivityRegularizer/MaximumMaximum8sequential_80/dense_80/ActivityRegularizer/Mean:output:0=sequential_80/dense_80/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 24
2sequential_80/dense_80/ActivityRegularizer/Maximum±
4sequential_80/dense_80/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<26
4sequential_80/dense_80/ActivityRegularizer/truediv/x
2sequential_80/dense_80/ActivityRegularizer/truedivRealDiv=sequential_80/dense_80/ActivityRegularizer/truediv/x:output:06sequential_80/dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 24
2sequential_80/dense_80/ActivityRegularizer/truedivÄ
.sequential_80/dense_80/ActivityRegularizer/LogLog6sequential_80/dense_80/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/Log©
0sequential_80/dense_80/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<22
0sequential_80/dense_80/ActivityRegularizer/mul/xû
.sequential_80/dense_80/ActivityRegularizer/mulMul9sequential_80/dense_80/ActivityRegularizer/mul/x:output:02sequential_80/dense_80/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/mul©
0sequential_80/dense_80/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_80/dense_80/ActivityRegularizer/sub/xÿ
.sequential_80/dense_80/ActivityRegularizer/subSub9sequential_80/dense_80/ActivityRegularizer/sub/x:output:06sequential_80/dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/subµ
6sequential_80/dense_80/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?28
6sequential_80/dense_80/ActivityRegularizer/truediv_1/x
4sequential_80/dense_80/ActivityRegularizer/truediv_1RealDiv?sequential_80/dense_80/ActivityRegularizer/truediv_1/x:output:02sequential_80/dense_80/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 26
4sequential_80/dense_80/ActivityRegularizer/truediv_1Ê
0sequential_80/dense_80/ActivityRegularizer/Log_1Log8sequential_80/dense_80/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 22
0sequential_80/dense_80/ActivityRegularizer/Log_1­
2sequential_80/dense_80/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?24
2sequential_80/dense_80/ActivityRegularizer/mul_1/x
0sequential_80/dense_80/ActivityRegularizer/mul_1Mul;sequential_80/dense_80/ActivityRegularizer/mul_1/x:output:04sequential_80/dense_80/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 22
0sequential_80/dense_80/ActivityRegularizer/mul_1ø
.sequential_80/dense_80/ActivityRegularizer/addAddV22sequential_80/dense_80/ActivityRegularizer/mul:z:04sequential_80/dense_80/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/add®
0sequential_80/dense_80/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_80/dense_80/ActivityRegularizer/Const÷
.sequential_80/dense_80/ActivityRegularizer/SumSum2sequential_80/dense_80/ActivityRegularizer/add:z:09sequential_80/dense_80/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_80/dense_80/ActivityRegularizer/Sum­
2sequential_80/dense_80/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_80/dense_80/ActivityRegularizer/mul_2/x
0sequential_80/dense_80/ActivityRegularizer/mul_2Mul;sequential_80/dense_80/ActivityRegularizer/mul_2/x:output:07sequential_80/dense_80/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_80/dense_80/ActivityRegularizer/mul_2¶
0sequential_80/dense_80/ActivityRegularizer/ShapeShape"sequential_80/dense_80/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_80/dense_80/ActivityRegularizer/ShapeÊ
>sequential_80/dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_80/dense_80/ActivityRegularizer/strided_slice/stackÎ
@sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1Î
@sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2ä
8sequential_80/dense_80/ActivityRegularizer/strided_sliceStridedSlice9sequential_80/dense_80/ActivityRegularizer/Shape:output:0Gsequential_80/dense_80/ActivityRegularizer/strided_slice/stack:output:0Isequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_80/dense_80/ActivityRegularizer/strided_sliceÝ
/sequential_80/dense_80/ActivityRegularizer/CastCastAsequential_80/dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_80/dense_80/ActivityRegularizer/Cast
4sequential_80/dense_80/ActivityRegularizer/truediv_2RealDiv4sequential_80/dense_80/ActivityRegularizer/mul_2:z:03sequential_80/dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_80/dense_80/ActivityRegularizer/truediv_2Ò
,sequential_81/dense_81/MatMul/ReadVariableOpReadVariableOp5sequential_81_dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02.
,sequential_81/dense_81/MatMul/ReadVariableOpÔ
sequential_81/dense_81/MatMulMatMul"sequential_80/dense_80/Sigmoid:y:04sequential_81/dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
sequential_81/dense_81/MatMulÑ
-sequential_81/dense_81/BiasAdd/ReadVariableOpReadVariableOp6sequential_81_dense_81_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02/
-sequential_81/dense_81/BiasAdd/ReadVariableOpÝ
sequential_81/dense_81/BiasAddBiasAdd'sequential_81/dense_81/MatMul:product:05sequential_81/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_81/dense_81/BiasAdd¦
sequential_81/dense_81/SigmoidSigmoid'sequential_81/dense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_81/dense_81/SigmoidÜ
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_80_dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulÜ
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_81_dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mul
IdentityIdentity"sequential_81/dense_81/Sigmoid:y:02^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp.^sequential_80/dense_80/BiasAdd/ReadVariableOp-^sequential_80/dense_80/MatMul/ReadVariableOp.^sequential_81/dense_81/BiasAdd/ReadVariableOp-^sequential_81/dense_81/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity¥

Identity_1Identity8sequential_80/dense_80/ActivityRegularizer/truediv_2:z:02^dense_80/kernel/Regularizer/Square/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp.^sequential_80/dense_80/BiasAdd/ReadVariableOp-^sequential_80/dense_80/MatMul/ReadVariableOp.^sequential_81/dense_81/BiasAdd/ReadVariableOp-^sequential_81/dense_81/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_80/dense_80/BiasAdd/ReadVariableOp-sequential_80/dense_80/BiasAdd/ReadVariableOp2\
,sequential_80/dense_80/MatMul/ReadVariableOp,sequential_80/dense_80/MatMul/ReadVariableOp2^
-sequential_81/dense_81/BiasAdd/ReadVariableOp-sequential_81/dense_81/BiasAdd/ReadVariableOp2\
,sequential_81/dense_81/MatMul/ReadVariableOp,sequential_81/dense_81/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
²A
ä
K__inference_sequential_80_layer_call_and_return_conditional_losses_16626521

inputs9
'dense_80_matmul_readvariableop_resource:^ 6
(dense_80_biasadd_readvariableop_resource: 
identity

identity_1¢dense_80/BiasAdd/ReadVariableOp¢dense_80/MatMul/ReadVariableOp¢1dense_80/kernel/Regularizer/Square/ReadVariableOp¨
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02 
dense_80/MatMul/ReadVariableOp
dense_80/MatMulMatMulinputs&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_80/MatMul§
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_80/BiasAdd/ReadVariableOp¥
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_80/BiasAdd|
dense_80/SigmoidSigmoiddense_80/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_80/Sigmoid¬
3dense_80/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_80/ActivityRegularizer/Mean/reduction_indicesÇ
!dense_80/ActivityRegularizer/MeanMeandense_80/Sigmoid:y:0<dense_80/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2#
!dense_80/ActivityRegularizer/Mean
&dense_80/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2(
&dense_80/ActivityRegularizer/Maximum/yÙ
$dense_80/ActivityRegularizer/MaximumMaximum*dense_80/ActivityRegularizer/Mean:output:0/dense_80/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2&
$dense_80/ActivityRegularizer/Maximum
&dense_80/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&dense_80/ActivityRegularizer/truediv/x×
$dense_80/ActivityRegularizer/truedivRealDiv/dense_80/ActivityRegularizer/truediv/x:output:0(dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2&
$dense_80/ActivityRegularizer/truediv
 dense_80/ActivityRegularizer/LogLog(dense_80/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/Log
"dense_80/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"dense_80/ActivityRegularizer/mul/xÃ
 dense_80/ActivityRegularizer/mulMul+dense_80/ActivityRegularizer/mul/x:output:0$dense_80/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/mul
"dense_80/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dense_80/ActivityRegularizer/sub/xÇ
 dense_80/ActivityRegularizer/subSub+dense_80/ActivityRegularizer/sub/x:output:0(dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/sub
(dense_80/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2*
(dense_80/ActivityRegularizer/truediv_1/xÙ
&dense_80/ActivityRegularizer/truediv_1RealDiv1dense_80/ActivityRegularizer/truediv_1/x:output:0$dense_80/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2(
&dense_80/ActivityRegularizer/truediv_1 
"dense_80/ActivityRegularizer/Log_1Log*dense_80/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2$
"dense_80/ActivityRegularizer/Log_1
$dense_80/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2&
$dense_80/ActivityRegularizer/mul_1/xË
"dense_80/ActivityRegularizer/mul_1Mul-dense_80/ActivityRegularizer/mul_1/x:output:0&dense_80/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2$
"dense_80/ActivityRegularizer/mul_1À
 dense_80/ActivityRegularizer/addAddV2$dense_80/ActivityRegularizer/mul:z:0&dense_80/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/add
"dense_80/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_80/ActivityRegularizer/Const¿
 dense_80/ActivityRegularizer/SumSum$dense_80/ActivityRegularizer/add:z:0+dense_80/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_80/ActivityRegularizer/Sum
$dense_80/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dense_80/ActivityRegularizer/mul_2/xÊ
"dense_80/ActivityRegularizer/mul_2Mul-dense_80/ActivityRegularizer/mul_2/x:output:0)dense_80/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_80/ActivityRegularizer/mul_2
"dense_80/ActivityRegularizer/ShapeShapedense_80/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_80/ActivityRegularizer/Shape®
0dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_80/ActivityRegularizer/strided_slice/stack²
2dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_1²
2dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_80/ActivityRegularizer/strided_slice/stack_2
*dense_80/ActivityRegularizer/strided_sliceStridedSlice+dense_80/ActivityRegularizer/Shape:output:09dense_80/ActivityRegularizer/strided_slice/stack:output:0;dense_80/ActivityRegularizer/strided_slice/stack_1:output:0;dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_80/ActivityRegularizer/strided_slice³
!dense_80/ActivityRegularizer/CastCast3dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_80/ActivityRegularizer/CastË
&dense_80/ActivityRegularizer/truediv_2RealDiv&dense_80/ActivityRegularizer/mul_2:z:0%dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_80/ActivityRegularizer/truediv_2Î
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulß
IdentityIdentitydense_80/Sigmoid:y:0 ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityè

Identity_1Identity*dense_80/ActivityRegularizer/truediv_2:z:0 ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
Î
Ê
&__inference_signature_wrapper_16626257
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
#__inference__wrapped_model_166257272
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
ó
Ï
1__inference_autoencoder_40_layer_call_fn_16626271
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
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_166260922
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
²

0__inference_sequential_80_layer_call_fn_16625810
input_41
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_41unknown	unknown_0*
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
K__inference_sequential_80_layer_call_and_return_conditional_losses_166258022
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
input_41
³
R
2__inference_dense_80_activity_regularizer_16625756

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
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626614
dense_81_input9
'dense_81_matmul_readvariableop_resource: ^6
(dense_81_biasadd_readvariableop_resource:^
identity¢dense_81/BiasAdd/ReadVariableOp¢dense_81/MatMul/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¨
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_81/MatMul/ReadVariableOp
dense_81/MatMulMatMuldense_81_input&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/MatMul§
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_81/BiasAdd/ReadVariableOp¥
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/BiasAdd|
dense_81/SigmoidSigmoiddense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/SigmoidÎ
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulß
IdentityIdentitydense_81/Sigmoid:y:0 ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_81_input

Õ
1__inference_autoencoder_40_layer_call_fn_16626174
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
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_166261482
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
Ç
Ü
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626631
dense_81_input9
'dense_81_matmul_readvariableop_resource: ^6
(dense_81_biasadd_readvariableop_resource:^
identity¢dense_81/BiasAdd/ReadVariableOp¢dense_81/MatMul/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp¨
dense_81/MatMul/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_81/MatMul/ReadVariableOp
dense_81/MatMulMatMuldense_81_input&dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/MatMul§
dense_81/BiasAdd/ReadVariableOpReadVariableOp(dense_81_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_81/BiasAdd/ReadVariableOp¥
dense_81/BiasAddBiasAdddense_81/MatMul:product:0'dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/BiasAdd|
dense_81/SigmoidSigmoiddense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_81/SigmoidÎ
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulß
IdentityIdentitydense_81/Sigmoid:y:0 ^dense_81/BiasAdd/ReadVariableOp^dense_81/MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_81/BiasAdd/ReadVariableOpdense_81/BiasAdd/ReadVariableOp2@
dense_81/MatMul/ReadVariableOpdense_81/MatMul/ReadVariableOp2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_81_input
ä]

#__inference__wrapped_model_16625727
input_1V
Dautoencoder_40_sequential_80_dense_80_matmul_readvariableop_resource:^ S
Eautoencoder_40_sequential_80_dense_80_biasadd_readvariableop_resource: V
Dautoencoder_40_sequential_81_dense_81_matmul_readvariableop_resource: ^S
Eautoencoder_40_sequential_81_dense_81_biasadd_readvariableop_resource:^
identity¢<autoencoder_40/sequential_80/dense_80/BiasAdd/ReadVariableOp¢;autoencoder_40/sequential_80/dense_80/MatMul/ReadVariableOp¢<autoencoder_40/sequential_81/dense_81/BiasAdd/ReadVariableOp¢;autoencoder_40/sequential_81/dense_81/MatMul/ReadVariableOpÿ
;autoencoder_40/sequential_80/dense_80/MatMul/ReadVariableOpReadVariableOpDautoencoder_40_sequential_80_dense_80_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02=
;autoencoder_40/sequential_80/dense_80/MatMul/ReadVariableOpæ
,autoencoder_40/sequential_80/dense_80/MatMulMatMulinput_1Cautoencoder_40/sequential_80/dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,autoencoder_40/sequential_80/dense_80/MatMulþ
<autoencoder_40/sequential_80/dense_80/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_40_sequential_80_dense_80_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<autoencoder_40/sequential_80/dense_80/BiasAdd/ReadVariableOp
-autoencoder_40/sequential_80/dense_80/BiasAddBiasAdd6autoencoder_40/sequential_80/dense_80/MatMul:product:0Dautoencoder_40/sequential_80/dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-autoencoder_40/sequential_80/dense_80/BiasAddÓ
-autoencoder_40/sequential_80/dense_80/SigmoidSigmoid6autoencoder_40/sequential_80/dense_80/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-autoencoder_40/sequential_80/dense_80/Sigmoidæ
Pautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2R
Pautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Mean/reduction_indices»
>autoencoder_40/sequential_80/dense_80/ActivityRegularizer/MeanMean1autoencoder_40/sequential_80/dense_80/Sigmoid:y:0Yautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2@
>autoencoder_40/sequential_80/dense_80/ActivityRegularizer/MeanÏ
Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2E
Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Maximum/yÍ
Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/MaximumMaximumGautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Mean:output:0Lautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/MaximumÏ
Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2E
Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv/xË
Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truedivRealDivLautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv/x:output:0Eautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truedivñ
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/LogLogEautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2?
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/LogÇ
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2A
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul/x·
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/mulMulHautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul/x:output:0Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2?
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/mulÇ
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/sub/x»
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/subSubHautoencoder_40/sequential_80/dense_80/ActivityRegularizer/sub/x:output:0Eautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2?
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/subÓ
Eautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2G
Eautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv_1/xÍ
Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv_1RealDivNautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv_1/x:output:0Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv_1÷
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/Log_1LogGautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/Log_1Ë
Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2C
Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_1/x¿
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_1MulJautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_1/x:output:0Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2A
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_1´
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/addAddV2Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul:z:0Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2?
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/addÌ
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2A
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/Const³
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/SumSumAautoencoder_40/sequential_80/dense_80/ActivityRegularizer/add:z:0Hautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2?
=autoencoder_40/sequential_80/dense_80/ActivityRegularizer/SumË
Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2C
Aautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_2/x¾
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_2MulJautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_2/x:output:0Fautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_2ã
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/ShapeShape1autoencoder_40/sequential_80/dense_80/Sigmoid:y:0*
T0*
_output_shapes
:2A
?autoencoder_40/sequential_80/dense_80/ActivityRegularizer/Shapeè
Mautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stackì
Oautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1ì
Oautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2¾
Gautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_sliceStridedSliceHautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Shape:output:0Vautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stack:output:0Xautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_1:output:0Xautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice
>autoencoder_40/sequential_80/dense_80/ActivityRegularizer/CastCastPautoencoder_40/sequential_80/dense_80/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>autoencoder_40/sequential_80/dense_80/ActivityRegularizer/Cast¿
Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv_2RealDivCautoencoder_40/sequential_80/dense_80/ActivityRegularizer/mul_2:z:0Bautoencoder_40/sequential_80/dense_80/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2E
Cautoencoder_40/sequential_80/dense_80/ActivityRegularizer/truediv_2ÿ
;autoencoder_40/sequential_81/dense_81/MatMul/ReadVariableOpReadVariableOpDautoencoder_40_sequential_81_dense_81_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02=
;autoencoder_40/sequential_81/dense_81/MatMul/ReadVariableOp
,autoencoder_40/sequential_81/dense_81/MatMulMatMul1autoencoder_40/sequential_80/dense_80/Sigmoid:y:0Cautoencoder_40/sequential_81/dense_81/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2.
,autoencoder_40/sequential_81/dense_81/MatMulþ
<autoencoder_40/sequential_81/dense_81/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_40_sequential_81_dense_81_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02>
<autoencoder_40/sequential_81/dense_81/BiasAdd/ReadVariableOp
-autoencoder_40/sequential_81/dense_81/BiasAddBiasAdd6autoencoder_40/sequential_81/dense_81/MatMul:product:0Dautoencoder_40/sequential_81/dense_81/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2/
-autoencoder_40/sequential_81/dense_81/BiasAddÓ
-autoencoder_40/sequential_81/dense_81/SigmoidSigmoid6autoencoder_40/sequential_81/dense_81/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2/
-autoencoder_40/sequential_81/dense_81/Sigmoidÿ
IdentityIdentity1autoencoder_40/sequential_81/dense_81/Sigmoid:y:0=^autoencoder_40/sequential_80/dense_80/BiasAdd/ReadVariableOp<^autoencoder_40/sequential_80/dense_80/MatMul/ReadVariableOp=^autoencoder_40/sequential_81/dense_81/BiasAdd/ReadVariableOp<^autoencoder_40/sequential_81/dense_81/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2|
<autoencoder_40/sequential_80/dense_80/BiasAdd/ReadVariableOp<autoencoder_40/sequential_80/dense_80/BiasAdd/ReadVariableOp2z
;autoencoder_40/sequential_80/dense_80/MatMul/ReadVariableOp;autoencoder_40/sequential_80/dense_80/MatMul/ReadVariableOp2|
<autoencoder_40/sequential_81/dense_81/BiasAdd/ReadVariableOp<autoencoder_40/sequential_81/dense_81/BiasAdd/ReadVariableOp2z
;autoencoder_40/sequential_81/dense_81/MatMul/ReadVariableOp;autoencoder_40/sequential_81/dense_81/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
¬

0__inference_sequential_80_layer_call_fn_16626419

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
K__inference_sequential_80_layer_call_and_return_conditional_losses_166258022
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
²

0__inference_sequential_80_layer_call_fn_16625886
input_41
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_41unknown	unknown_0*
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
K__inference_sequential_80_layer_call_and_return_conditional_losses_166258682
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
input_41


K__inference_sequential_81_layer_call_and_return_conditional_losses_16626014

inputs#
dense_81_16626002: ^
dense_81_16626004:^
identity¢ dense_81/StatefulPartitionedCall¢1dense_81/kernel/Regularizer/Square/ReadVariableOp
 dense_81/StatefulPartitionedCallStatefulPartitionedCallinputsdense_81_16626002dense_81_16626004*
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
F__inference_dense_81_layer_call_and_return_conditional_losses_166259582"
 dense_81/StatefulPartitionedCall¸
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_81_16626002*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulÔ
IdentityIdentity)dense_81/StatefulPartitionedCall:output:0!^dense_81/StatefulPartitionedCall2^dense_81/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2D
 dense_81/StatefulPartitionedCall dense_81/StatefulPartitionedCall2f
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨
Ç
J__inference_dense_80_layer_call_and_return_all_conditional_losses_16626657

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
F__inference_dense_80_layer_call_and_return_conditional_losses_166257802
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
2__inference_dense_80_activity_regularizer_166257562
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
ä
³
__inference_loss_fn_0_16626668L
:dense_80_kernel_regularizer_square_readvariableop_resource:^ 
identity¢1dense_80/kernel/Regularizer/Square/ReadVariableOpá
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_80_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mul
IdentityIdentity#dense_80/kernel/Regularizer/mul:z:02^dense_80/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp
­
«
F__inference_dense_80_layer_call_and_return_conditional_losses_16625780

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_80/kernel/Regularizer/Square/ReadVariableOp
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
1dense_80/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_80/kernel/Regularizer/Square/ReadVariableOp¶
"dense_80/kernel/Regularizer/SquareSquare9dense_80/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_80/kernel/Regularizer/Square
!dense_80/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_80/kernel/Regularizer/Const¾
dense_80/kernel/Regularizer/SumSum&dense_80/kernel/Regularizer/Square:y:0*dense_80/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/Sum
!dense_80/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_80/kernel/Regularizer/mul/xÀ
dense_80/kernel/Regularizer/mulMul*dense_80/kernel/Regularizer/mul/x:output:0(dense_80/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_80/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_80/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_80/kernel/Regularizer/Square/ReadVariableOp1dense_80/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs


+__inference_dense_81_layer_call_fn_16626683

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
F__inference_dense_81_layer_call_and_return_conditional_losses_166259582
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
F__inference_dense_81_layer_call_and_return_conditional_losses_16626700

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_81/kernel/Regularizer/Square/ReadVariableOp
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
1dense_81/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_81/kernel/Regularizer/Square/ReadVariableOp¶
"dense_81/kernel/Regularizer/SquareSquare9dense_81/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_81/kernel/Regularizer/Square
!dense_81/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_81/kernel/Regularizer/Const¾
dense_81/kernel/Regularizer/SumSum&dense_81/kernel/Regularizer/Square:y:0*dense_81/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/Sum
!dense_81/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_81/kernel/Regularizer/mul/xÀ
dense_81/kernel/Regularizer/mulMul*dense_81/kernel/Regularizer/mul/x:output:0(dense_81/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_81/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_81/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_81/kernel/Regularizer/Square/ReadVariableOp1dense_81/kernel/Regularizer/Square/ReadVariableOp:O K
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
_tf_keras_model{"name": "autoencoder_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
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
_tf_keras_sequentialÞ{"name": "sequential_80", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_80", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_41"}}, {"class_name": "Dense", "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_41"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_80", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_41"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
¾
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_sequentialé{"name": "sequential_81", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_81", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_81_input"}}, {"class_name": "Dense", "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_81_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_81", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_81_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
{"name": "dense_80", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_80", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
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
_tf_keras_layer®{"name": "dense_81", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_81", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
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
!:^ 2dense_80/kernel
: 2dense_80/bias
!: ^2dense_81/kernel
:^2dense_81/bias
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
1__inference_autoencoder_40_layer_call_fn_16626104
1__inference_autoencoder_40_layer_call_fn_16626271
1__inference_autoencoder_40_layer_call_fn_16626285
1__inference_autoencoder_40_layer_call_fn_16626174®
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
#__inference__wrapped_model_16625727¶
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
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626344
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626403
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626202
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626230®
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
0__inference_sequential_80_layer_call_fn_16625810
0__inference_sequential_80_layer_call_fn_16626419
0__inference_sequential_80_layer_call_fn_16626429
0__inference_sequential_80_layer_call_fn_16625886À
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
K__inference_sequential_80_layer_call_and_return_conditional_losses_16626475
K__inference_sequential_80_layer_call_and_return_conditional_losses_16626521
K__inference_sequential_80_layer_call_and_return_conditional_losses_16625910
K__inference_sequential_80_layer_call_and_return_conditional_losses_16625934À
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
0__inference_sequential_81_layer_call_fn_16626536
0__inference_sequential_81_layer_call_fn_16626545
0__inference_sequential_81_layer_call_fn_16626554
0__inference_sequential_81_layer_call_fn_16626563À
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626580
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626597
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626614
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626631À
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
&__inference_signature_wrapper_16626257input_1"
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
+__inference_dense_80_layer_call_fn_16626646¢
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
J__inference_dense_80_layer_call_and_return_all_conditional_losses_16626657¢
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
__inference_loss_fn_0_16626668
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
+__inference_dense_81_layer_call_fn_16626683¢
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
F__inference_dense_81_layer_call_and_return_conditional_losses_16626700¢
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
__inference_loss_fn_1_16626711
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
2__inference_dense_80_activity_regularizer_16625756²
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
F__inference_dense_80_layer_call_and_return_conditional_losses_16626728¢
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
#__inference__wrapped_model_16625727m0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ^
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ^Á
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626202q4¢1
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
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626230q4¢1
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
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626344k.¢+
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
L__inference_autoencoder_40_layer_call_and_return_conditional_losses_16626403k.¢+
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
1__inference_autoencoder_40_layer_call_fn_16626104V4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p 
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_40_layer_call_fn_16626174V4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_40_layer_call_fn_16626271P.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p 
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_40_layer_call_fn_16626285P.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p
ª "ÿÿÿÿÿÿÿÿÿ^e
2__inference_dense_80_activity_regularizer_16625756/$¢!
¢


activation
ª " ¸
J__inference_dense_80_layer_call_and_return_all_conditional_losses_16626657j/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 ¦
F__inference_dense_80_layer_call_and_return_conditional_losses_16626728\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_dense_80_layer_call_fn_16626646O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "ÿÿÿÿÿÿÿÿÿ ¦
F__inference_dense_81_layer_call_and_return_conditional_losses_16626700\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 ~
+__inference_dense_81_layer_call_fn_16626683O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ^=
__inference_loss_fn_0_16626668¢

¢ 
ª " =
__inference_loss_fn_1_16626711¢

¢ 
ª " Ã
K__inference_sequential_80_layer_call_and_return_conditional_losses_16625910t9¢6
/¢,
"
input_41ÿÿÿÿÿÿÿÿÿ^
p 

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Ã
K__inference_sequential_80_layer_call_and_return_conditional_losses_16625934t9¢6
/¢,
"
input_41ÿÿÿÿÿÿÿÿÿ^
p

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Á
K__inference_sequential_80_layer_call_and_return_conditional_losses_16626475r7¢4
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
K__inference_sequential_80_layer_call_and_return_conditional_losses_16626521r7¢4
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
0__inference_sequential_80_layer_call_fn_16625810Y9¢6
/¢,
"
input_41ÿÿÿÿÿÿÿÿÿ^
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_80_layer_call_fn_16625886Y9¢6
/¢,
"
input_41ÿÿÿÿÿÿÿÿÿ^
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_80_layer_call_fn_16626419W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_80_layer_call_fn_16626429W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p

 
ª "ÿÿÿÿÿÿÿÿÿ ³
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626580d7¢4
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626597d7¢4
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
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626614l?¢<
5¢2
(%
dense_81_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 »
K__inference_sequential_81_layer_call_and_return_conditional_losses_16626631l?¢<
5¢2
(%
dense_81_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 
0__inference_sequential_81_layer_call_fn_16626536_?¢<
5¢2
(%
dense_81_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_81_layer_call_fn_16626545W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_81_layer_call_fn_16626554W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_81_layer_call_fn_16626563_?¢<
5¢2
(%
dense_81_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ^¢
&__inference_signature_wrapper_16626257x;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ^"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ^