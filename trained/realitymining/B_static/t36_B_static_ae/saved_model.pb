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
dense_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ * 
shared_namedense_70/kernel
s
#dense_70/kernel/Read/ReadVariableOpReadVariableOpdense_70/kernel*
_output_shapes

:^ *
dtype0
r
dense_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_70/bias
k
!dense_70/bias/Read/ReadVariableOpReadVariableOpdense_70/bias*
_output_shapes
: *
dtype0
z
dense_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^* 
shared_namedense_71/kernel
s
#dense_71/kernel/Read/ReadVariableOpReadVariableOpdense_71/kernel*
_output_shapes

: ^*
dtype0
r
dense_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_71/bias
k
!dense_71/bias/Read/ReadVariableOpReadVariableOpdense_71/bias*
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
VARIABLE_VALUEdense_70/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_70/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_71/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_71/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_70/kerneldense_70/biasdense_71/kerneldense_71/bias*
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
&__inference_signature_wrapper_16620002
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
±
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_70/kernel/Read/ReadVariableOp!dense_70/bias/Read/ReadVariableOp#dense_71/kernel/Read/ReadVariableOp!dense_71/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_16620508
Ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_70/kerneldense_70/biasdense_71/kerneldense_71/bias*
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
$__inference__traced_restore_16620530¥ô
Êe
º
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16620148
xG
5sequential_70_dense_70_matmul_readvariableop_resource:^ D
6sequential_70_dense_70_biasadd_readvariableop_resource: G
5sequential_71_dense_71_matmul_readvariableop_resource: ^D
6sequential_71_dense_71_biasadd_readvariableop_resource:^
identity

identity_1¢1dense_70/kernel/Regularizer/Square/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¢-sequential_70/dense_70/BiasAdd/ReadVariableOp¢,sequential_70/dense_70/MatMul/ReadVariableOp¢-sequential_71/dense_71/BiasAdd/ReadVariableOp¢,sequential_71/dense_71/MatMul/ReadVariableOpÒ
,sequential_70/dense_70/MatMul/ReadVariableOpReadVariableOp5sequential_70_dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02.
,sequential_70/dense_70/MatMul/ReadVariableOp³
sequential_70/dense_70/MatMulMatMulx4sequential_70/dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_70/dense_70/MatMulÑ
-sequential_70/dense_70/BiasAdd/ReadVariableOpReadVariableOp6sequential_70_dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_70/dense_70/BiasAdd/ReadVariableOpÝ
sequential_70/dense_70/BiasAddBiasAdd'sequential_70/dense_70/MatMul:product:05sequential_70/dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_70/dense_70/BiasAdd¦
sequential_70/dense_70/SigmoidSigmoid'sequential_70/dense_70/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_70/dense_70/SigmoidÈ
Asequential_70/dense_70/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_70/dense_70/ActivityRegularizer/Mean/reduction_indicesÿ
/sequential_70/dense_70/ActivityRegularizer/MeanMean"sequential_70/dense_70/Sigmoid:y:0Jsequential_70/dense_70/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 21
/sequential_70/dense_70/ActivityRegularizer/Mean±
4sequential_70/dense_70/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.26
4sequential_70/dense_70/ActivityRegularizer/Maximum/y
2sequential_70/dense_70/ActivityRegularizer/MaximumMaximum8sequential_70/dense_70/ActivityRegularizer/Mean:output:0=sequential_70/dense_70/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 24
2sequential_70/dense_70/ActivityRegularizer/Maximum±
4sequential_70/dense_70/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<26
4sequential_70/dense_70/ActivityRegularizer/truediv/x
2sequential_70/dense_70/ActivityRegularizer/truedivRealDiv=sequential_70/dense_70/ActivityRegularizer/truediv/x:output:06sequential_70/dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 24
2sequential_70/dense_70/ActivityRegularizer/truedivÄ
.sequential_70/dense_70/ActivityRegularizer/LogLog6sequential_70/dense_70/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/Log©
0sequential_70/dense_70/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<22
0sequential_70/dense_70/ActivityRegularizer/mul/xû
.sequential_70/dense_70/ActivityRegularizer/mulMul9sequential_70/dense_70/ActivityRegularizer/mul/x:output:02sequential_70/dense_70/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/mul©
0sequential_70/dense_70/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_70/dense_70/ActivityRegularizer/sub/xÿ
.sequential_70/dense_70/ActivityRegularizer/subSub9sequential_70/dense_70/ActivityRegularizer/sub/x:output:06sequential_70/dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/subµ
6sequential_70/dense_70/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?28
6sequential_70/dense_70/ActivityRegularizer/truediv_1/x
4sequential_70/dense_70/ActivityRegularizer/truediv_1RealDiv?sequential_70/dense_70/ActivityRegularizer/truediv_1/x:output:02sequential_70/dense_70/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 26
4sequential_70/dense_70/ActivityRegularizer/truediv_1Ê
0sequential_70/dense_70/ActivityRegularizer/Log_1Log8sequential_70/dense_70/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 22
0sequential_70/dense_70/ActivityRegularizer/Log_1­
2sequential_70/dense_70/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?24
2sequential_70/dense_70/ActivityRegularizer/mul_1/x
0sequential_70/dense_70/ActivityRegularizer/mul_1Mul;sequential_70/dense_70/ActivityRegularizer/mul_1/x:output:04sequential_70/dense_70/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 22
0sequential_70/dense_70/ActivityRegularizer/mul_1ø
.sequential_70/dense_70/ActivityRegularizer/addAddV22sequential_70/dense_70/ActivityRegularizer/mul:z:04sequential_70/dense_70/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/add®
0sequential_70/dense_70/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_70/dense_70/ActivityRegularizer/Const÷
.sequential_70/dense_70/ActivityRegularizer/SumSum2sequential_70/dense_70/ActivityRegularizer/add:z:09sequential_70/dense_70/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/Sum­
2sequential_70/dense_70/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_70/dense_70/ActivityRegularizer/mul_2/x
0sequential_70/dense_70/ActivityRegularizer/mul_2Mul;sequential_70/dense_70/ActivityRegularizer/mul_2/x:output:07sequential_70/dense_70/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_70/dense_70/ActivityRegularizer/mul_2¶
0sequential_70/dense_70/ActivityRegularizer/ShapeShape"sequential_70/dense_70/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_70/dense_70/ActivityRegularizer/ShapeÊ
>sequential_70/dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_70/dense_70/ActivityRegularizer/strided_slice/stackÎ
@sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1Î
@sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2ä
8sequential_70/dense_70/ActivityRegularizer/strided_sliceStridedSlice9sequential_70/dense_70/ActivityRegularizer/Shape:output:0Gsequential_70/dense_70/ActivityRegularizer/strided_slice/stack:output:0Isequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_70/dense_70/ActivityRegularizer/strided_sliceÝ
/sequential_70/dense_70/ActivityRegularizer/CastCastAsequential_70/dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_70/dense_70/ActivityRegularizer/Cast
4sequential_70/dense_70/ActivityRegularizer/truediv_2RealDiv4sequential_70/dense_70/ActivityRegularizer/mul_2:z:03sequential_70/dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_70/dense_70/ActivityRegularizer/truediv_2Ò
,sequential_71/dense_71/MatMul/ReadVariableOpReadVariableOp5sequential_71_dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02.
,sequential_71/dense_71/MatMul/ReadVariableOpÔ
sequential_71/dense_71/MatMulMatMul"sequential_70/dense_70/Sigmoid:y:04sequential_71/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
sequential_71/dense_71/MatMulÑ
-sequential_71/dense_71/BiasAdd/ReadVariableOpReadVariableOp6sequential_71_dense_71_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02/
-sequential_71/dense_71/BiasAdd/ReadVariableOpÝ
sequential_71/dense_71/BiasAddBiasAdd'sequential_71/dense_71/MatMul:product:05sequential_71/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_71/dense_71/BiasAdd¦
sequential_71/dense_71/SigmoidSigmoid'sequential_71/dense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_71/dense_71/SigmoidÜ
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_70_dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulÜ
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_71_dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mul
IdentityIdentity"sequential_71/dense_71/Sigmoid:y:02^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp.^sequential_70/dense_70/BiasAdd/ReadVariableOp-^sequential_70/dense_70/MatMul/ReadVariableOp.^sequential_71/dense_71/BiasAdd/ReadVariableOp-^sequential_71/dense_71/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity¥

Identity_1Identity8sequential_70/dense_70/ActivityRegularizer/truediv_2:z:02^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp.^sequential_70/dense_70/BiasAdd/ReadVariableOp-^sequential_70/dense_70/MatMul/ReadVariableOp.^sequential_71/dense_71/BiasAdd/ReadVariableOp-^sequential_71/dense_71/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_70/dense_70/BiasAdd/ReadVariableOp-sequential_70/dense_70/BiasAdd/ReadVariableOp2\
,sequential_70/dense_70/MatMul/ReadVariableOp,sequential_70/dense_70/MatMul/ReadVariableOp2^
-sequential_71/dense_71/BiasAdd/ReadVariableOp-sequential_71/dense_71/BiasAdd/ReadVariableOp2\
,sequential_71/dense_71/MatMul/ReadVariableOp,sequential_71/dense_71/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
ä
³
__inference_loss_fn_1_16620456L
:dense_71_kernel_regularizer_square_readvariableop_resource: ^
identity¢1dense_71/kernel/Regularizer/Square/ReadVariableOpá
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_71_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mul
IdentityIdentity#dense_71/kernel/Regularizer/mul:z:02^dense_71/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp
ò$
Î
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16619837
x(
sequential_70_16619812:^ $
sequential_70_16619814: (
sequential_71_16619818: ^$
sequential_71_16619820:^
identity

identity_1¢1dense_70/kernel/Regularizer/Square/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¢%sequential_70/StatefulPartitionedCall¢%sequential_71/StatefulPartitionedCall±
%sequential_70/StatefulPartitionedCallStatefulPartitionedCallxsequential_70_16619812sequential_70_16619814*
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_166195472'
%sequential_70/StatefulPartitionedCallÛ
%sequential_71/StatefulPartitionedCallStatefulPartitionedCall.sequential_70/StatefulPartitionedCall:output:0sequential_71_16619818sequential_71_16619820*
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_166197162'
%sequential_71/StatefulPartitionedCall½
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_70_16619812*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mul½
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_71_16619818*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulº
IdentityIdentity.sequential_71/StatefulPartitionedCall:output:02^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp&^sequential_70/StatefulPartitionedCall&^sequential_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_70/StatefulPartitionedCall:output:12^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp&^sequential_70/StatefulPartitionedCall&^sequential_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_70/StatefulPartitionedCall%sequential_70/StatefulPartitionedCall2N
%sequential_71/StatefulPartitionedCall%sequential_71/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
ô"

K__inference_sequential_70_layer_call_and_return_conditional_losses_16619679
input_36#
dense_70_16619658:^ 
dense_70_16619660: 
identity

identity_1¢ dense_70/StatefulPartitionedCall¢1dense_70/kernel/Regularizer/Square/ReadVariableOp
 dense_70/StatefulPartitionedCallStatefulPartitionedCallinput_36dense_70_16619658dense_70_16619660*
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
F__inference_dense_70_layer_call_and_return_conditional_losses_166195252"
 dense_70/StatefulPartitionedCallü
,dense_70/ActivityRegularizer/PartitionedCallPartitionedCall)dense_70/StatefulPartitionedCall:output:0*
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
2__inference_dense_70_activity_regularizer_166195012.
,dense_70/ActivityRegularizer/PartitionedCall¡
"dense_70/ActivityRegularizer/ShapeShape)dense_70/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_70/ActivityRegularizer/Shape®
0dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_70/ActivityRegularizer/strided_slice/stack²
2dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_1²
2dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_2
*dense_70/ActivityRegularizer/strided_sliceStridedSlice+dense_70/ActivityRegularizer/Shape:output:09dense_70/ActivityRegularizer/strided_slice/stack:output:0;dense_70/ActivityRegularizer/strided_slice/stack_1:output:0;dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_70/ActivityRegularizer/strided_slice³
!dense_70/ActivityRegularizer/CastCast3dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_70/ActivityRegularizer/CastÖ
$dense_70/ActivityRegularizer/truedivRealDiv5dense_70/ActivityRegularizer/PartitionedCall:output:0%dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_70/ActivityRegularizer/truediv¸
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_70_16619658*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulÔ
IdentityIdentity)dense_70/StatefulPartitionedCall:output:0!^dense_70/StatefulPartitionedCall2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_70/ActivityRegularizer/truediv:z:0!^dense_70/StatefulPartitionedCall2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_36
Á
¥
0__inference_sequential_71_layer_call_fn_16620281
dense_71_input
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_71_inputunknown	unknown_0*
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_166197162
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
_user_specified_namedense_71_input
%
Ô
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16619947
input_1(
sequential_70_16619922:^ $
sequential_70_16619924: (
sequential_71_16619928: ^$
sequential_71_16619930:^
identity

identity_1¢1dense_70/kernel/Regularizer/Square/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¢%sequential_70/StatefulPartitionedCall¢%sequential_71/StatefulPartitionedCall·
%sequential_70/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_70_16619922sequential_70_16619924*
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_166195472'
%sequential_70/StatefulPartitionedCallÛ
%sequential_71/StatefulPartitionedCallStatefulPartitionedCall.sequential_70/StatefulPartitionedCall:output:0sequential_71_16619928sequential_71_16619930*
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_166197162'
%sequential_71/StatefulPartitionedCall½
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_70_16619922*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mul½
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_71_16619928*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulº
IdentityIdentity.sequential_71/StatefulPartitionedCall:output:02^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp&^sequential_70/StatefulPartitionedCall&^sequential_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_70/StatefulPartitionedCall:output:12^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp&^sequential_70/StatefulPartitionedCall&^sequential_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_70/StatefulPartitionedCall%sequential_70/StatefulPartitionedCall2N
%sequential_71/StatefulPartitionedCall%sequential_71/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
²

0__inference_sequential_70_layer_call_fn_16619631
input_36
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_36unknown	unknown_0*
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_166196132
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
input_36
²A
ä
K__inference_sequential_70_layer_call_and_return_conditional_losses_16620266

inputs9
'dense_70_matmul_readvariableop_resource:^ 6
(dense_70_biasadd_readvariableop_resource: 
identity

identity_1¢dense_70/BiasAdd/ReadVariableOp¢dense_70/MatMul/ReadVariableOp¢1dense_70/kernel/Regularizer/Square/ReadVariableOp¨
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02 
dense_70/MatMul/ReadVariableOp
dense_70/MatMulMatMulinputs&dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_70/MatMul§
dense_70/BiasAdd/ReadVariableOpReadVariableOp(dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_70/BiasAdd/ReadVariableOp¥
dense_70/BiasAddBiasAdddense_70/MatMul:product:0'dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_70/BiasAdd|
dense_70/SigmoidSigmoiddense_70/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_70/Sigmoid¬
3dense_70/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_70/ActivityRegularizer/Mean/reduction_indicesÇ
!dense_70/ActivityRegularizer/MeanMeandense_70/Sigmoid:y:0<dense_70/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2#
!dense_70/ActivityRegularizer/Mean
&dense_70/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2(
&dense_70/ActivityRegularizer/Maximum/yÙ
$dense_70/ActivityRegularizer/MaximumMaximum*dense_70/ActivityRegularizer/Mean:output:0/dense_70/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2&
$dense_70/ActivityRegularizer/Maximum
&dense_70/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&dense_70/ActivityRegularizer/truediv/x×
$dense_70/ActivityRegularizer/truedivRealDiv/dense_70/ActivityRegularizer/truediv/x:output:0(dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2&
$dense_70/ActivityRegularizer/truediv
 dense_70/ActivityRegularizer/LogLog(dense_70/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/Log
"dense_70/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"dense_70/ActivityRegularizer/mul/xÃ
 dense_70/ActivityRegularizer/mulMul+dense_70/ActivityRegularizer/mul/x:output:0$dense_70/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/mul
"dense_70/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dense_70/ActivityRegularizer/sub/xÇ
 dense_70/ActivityRegularizer/subSub+dense_70/ActivityRegularizer/sub/x:output:0(dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/sub
(dense_70/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2*
(dense_70/ActivityRegularizer/truediv_1/xÙ
&dense_70/ActivityRegularizer/truediv_1RealDiv1dense_70/ActivityRegularizer/truediv_1/x:output:0$dense_70/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2(
&dense_70/ActivityRegularizer/truediv_1 
"dense_70/ActivityRegularizer/Log_1Log*dense_70/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2$
"dense_70/ActivityRegularizer/Log_1
$dense_70/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2&
$dense_70/ActivityRegularizer/mul_1/xË
"dense_70/ActivityRegularizer/mul_1Mul-dense_70/ActivityRegularizer/mul_1/x:output:0&dense_70/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2$
"dense_70/ActivityRegularizer/mul_1À
 dense_70/ActivityRegularizer/addAddV2$dense_70/ActivityRegularizer/mul:z:0&dense_70/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/add
"dense_70/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_70/ActivityRegularizer/Const¿
 dense_70/ActivityRegularizer/SumSum$dense_70/ActivityRegularizer/add:z:0+dense_70/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/Sum
$dense_70/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dense_70/ActivityRegularizer/mul_2/xÊ
"dense_70/ActivityRegularizer/mul_2Mul-dense_70/ActivityRegularizer/mul_2/x:output:0)dense_70/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_70/ActivityRegularizer/mul_2
"dense_70/ActivityRegularizer/ShapeShapedense_70/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_70/ActivityRegularizer/Shape®
0dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_70/ActivityRegularizer/strided_slice/stack²
2dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_1²
2dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_2
*dense_70/ActivityRegularizer/strided_sliceStridedSlice+dense_70/ActivityRegularizer/Shape:output:09dense_70/ActivityRegularizer/strided_slice/stack:output:0;dense_70/ActivityRegularizer/strided_slice/stack_1:output:0;dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_70/ActivityRegularizer/strided_slice³
!dense_70/ActivityRegularizer/CastCast3dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_70/ActivityRegularizer/CastË
&dense_70/ActivityRegularizer/truediv_2RealDiv&dense_70/ActivityRegularizer/mul_2:z:0%dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_70/ActivityRegularizer/truediv_2Î
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulß
IdentityIdentitydense_70/Sigmoid:y:0 ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityè

Identity_1Identity*dense_70/ActivityRegularizer/truediv_2:z:0 ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2B
dense_70/BiasAdd/ReadVariableOpdense_70/BiasAdd/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
­
«
F__inference_dense_70_layer_call_and_return_conditional_losses_16619525

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_70/kernel/Regularizer/Square/ReadVariableOp
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
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
î"

K__inference_sequential_70_layer_call_and_return_conditional_losses_16619547

inputs#
dense_70_16619526:^ 
dense_70_16619528: 
identity

identity_1¢ dense_70/StatefulPartitionedCall¢1dense_70/kernel/Regularizer/Square/ReadVariableOp
 dense_70/StatefulPartitionedCallStatefulPartitionedCallinputsdense_70_16619526dense_70_16619528*
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
F__inference_dense_70_layer_call_and_return_conditional_losses_166195252"
 dense_70/StatefulPartitionedCallü
,dense_70/ActivityRegularizer/PartitionedCallPartitionedCall)dense_70/StatefulPartitionedCall:output:0*
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
2__inference_dense_70_activity_regularizer_166195012.
,dense_70/ActivityRegularizer/PartitionedCall¡
"dense_70/ActivityRegularizer/ShapeShape)dense_70/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_70/ActivityRegularizer/Shape®
0dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_70/ActivityRegularizer/strided_slice/stack²
2dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_1²
2dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_2
*dense_70/ActivityRegularizer/strided_sliceStridedSlice+dense_70/ActivityRegularizer/Shape:output:09dense_70/ActivityRegularizer/strided_slice/stack:output:0;dense_70/ActivityRegularizer/strided_slice/stack_1:output:0;dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_70/ActivityRegularizer/strided_slice³
!dense_70/ActivityRegularizer/CastCast3dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_70/ActivityRegularizer/CastÖ
$dense_70/ActivityRegularizer/truedivRealDiv5dense_70/ActivityRegularizer/PartitionedCall:output:0%dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_70/ActivityRegularizer/truediv¸
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_70_16619526*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulÔ
IdentityIdentity)dense_70/StatefulPartitionedCall:output:0!^dense_70/StatefulPartitionedCall2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_70/ActivityRegularizer/truediv:z:0!^dense_70/StatefulPartitionedCall2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
ä]

#__inference__wrapped_model_16619472
input_1V
Dautoencoder_35_sequential_70_dense_70_matmul_readvariableop_resource:^ S
Eautoencoder_35_sequential_70_dense_70_biasadd_readvariableop_resource: V
Dautoencoder_35_sequential_71_dense_71_matmul_readvariableop_resource: ^S
Eautoencoder_35_sequential_71_dense_71_biasadd_readvariableop_resource:^
identity¢<autoencoder_35/sequential_70/dense_70/BiasAdd/ReadVariableOp¢;autoencoder_35/sequential_70/dense_70/MatMul/ReadVariableOp¢<autoencoder_35/sequential_71/dense_71/BiasAdd/ReadVariableOp¢;autoencoder_35/sequential_71/dense_71/MatMul/ReadVariableOpÿ
;autoencoder_35/sequential_70/dense_70/MatMul/ReadVariableOpReadVariableOpDautoencoder_35_sequential_70_dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02=
;autoencoder_35/sequential_70/dense_70/MatMul/ReadVariableOpæ
,autoencoder_35/sequential_70/dense_70/MatMulMatMulinput_1Cautoencoder_35/sequential_70/dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,autoencoder_35/sequential_70/dense_70/MatMulþ
<autoencoder_35/sequential_70/dense_70/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_35_sequential_70_dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<autoencoder_35/sequential_70/dense_70/BiasAdd/ReadVariableOp
-autoencoder_35/sequential_70/dense_70/BiasAddBiasAdd6autoencoder_35/sequential_70/dense_70/MatMul:product:0Dautoencoder_35/sequential_70/dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-autoencoder_35/sequential_70/dense_70/BiasAddÓ
-autoencoder_35/sequential_70/dense_70/SigmoidSigmoid6autoencoder_35/sequential_70/dense_70/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-autoencoder_35/sequential_70/dense_70/Sigmoidæ
Pautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2R
Pautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Mean/reduction_indices»
>autoencoder_35/sequential_70/dense_70/ActivityRegularizer/MeanMean1autoencoder_35/sequential_70/dense_70/Sigmoid:y:0Yautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2@
>autoencoder_35/sequential_70/dense_70/ActivityRegularizer/MeanÏ
Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2E
Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Maximum/yÍ
Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/MaximumMaximumGautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Mean:output:0Lautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/MaximumÏ
Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2E
Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv/xË
Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truedivRealDivLautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv/x:output:0Eautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truedivñ
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/LogLogEautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2?
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/LogÇ
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2A
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul/x·
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/mulMulHautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul/x:output:0Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2?
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/mulÇ
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/sub/x»
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/subSubHautoencoder_35/sequential_70/dense_70/ActivityRegularizer/sub/x:output:0Eautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2?
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/subÓ
Eautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2G
Eautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv_1/xÍ
Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv_1RealDivNautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv_1/x:output:0Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv_1÷
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/Log_1LogGautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/Log_1Ë
Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2C
Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_1/x¿
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_1MulJautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_1/x:output:0Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2A
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_1´
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/addAddV2Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul:z:0Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2?
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/addÌ
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2A
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/Const³
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/SumSumAautoencoder_35/sequential_70/dense_70/ActivityRegularizer/add:z:0Hautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2?
=autoencoder_35/sequential_70/dense_70/ActivityRegularizer/SumË
Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2C
Aautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_2/x¾
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_2MulJautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_2/x:output:0Fautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_2ã
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/ShapeShape1autoencoder_35/sequential_70/dense_70/Sigmoid:y:0*
T0*
_output_shapes
:2A
?autoencoder_35/sequential_70/dense_70/ActivityRegularizer/Shapeè
Mautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stackì
Oautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1ì
Oautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2¾
Gautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_sliceStridedSliceHautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Shape:output:0Vautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stack:output:0Xautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1:output:0Xautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice
>autoencoder_35/sequential_70/dense_70/ActivityRegularizer/CastCastPautoencoder_35/sequential_70/dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>autoencoder_35/sequential_70/dense_70/ActivityRegularizer/Cast¿
Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv_2RealDivCautoencoder_35/sequential_70/dense_70/ActivityRegularizer/mul_2:z:0Bautoencoder_35/sequential_70/dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2E
Cautoencoder_35/sequential_70/dense_70/ActivityRegularizer/truediv_2ÿ
;autoencoder_35/sequential_71/dense_71/MatMul/ReadVariableOpReadVariableOpDautoencoder_35_sequential_71_dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02=
;autoencoder_35/sequential_71/dense_71/MatMul/ReadVariableOp
,autoencoder_35/sequential_71/dense_71/MatMulMatMul1autoencoder_35/sequential_70/dense_70/Sigmoid:y:0Cautoencoder_35/sequential_71/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2.
,autoencoder_35/sequential_71/dense_71/MatMulþ
<autoencoder_35/sequential_71/dense_71/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_35_sequential_71_dense_71_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02>
<autoencoder_35/sequential_71/dense_71/BiasAdd/ReadVariableOp
-autoencoder_35/sequential_71/dense_71/BiasAddBiasAdd6autoencoder_35/sequential_71/dense_71/MatMul:product:0Dautoencoder_35/sequential_71/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2/
-autoencoder_35/sequential_71/dense_71/BiasAddÓ
-autoencoder_35/sequential_71/dense_71/SigmoidSigmoid6autoencoder_35/sequential_71/dense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2/
-autoencoder_35/sequential_71/dense_71/Sigmoidÿ
IdentityIdentity1autoencoder_35/sequential_71/dense_71/Sigmoid:y:0=^autoencoder_35/sequential_70/dense_70/BiasAdd/ReadVariableOp<^autoencoder_35/sequential_70/dense_70/MatMul/ReadVariableOp=^autoencoder_35/sequential_71/dense_71/BiasAdd/ReadVariableOp<^autoencoder_35/sequential_71/dense_71/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2|
<autoencoder_35/sequential_70/dense_70/BiasAdd/ReadVariableOp<autoencoder_35/sequential_70/dense_70/BiasAdd/ReadVariableOp2z
;autoencoder_35/sequential_70/dense_70/MatMul/ReadVariableOp;autoencoder_35/sequential_70/dense_70/MatMul/ReadVariableOp2|
<autoencoder_35/sequential_71/dense_71/BiasAdd/ReadVariableOp<autoencoder_35/sequential_71/dense_71/BiasAdd/ReadVariableOp2z
;autoencoder_35/sequential_71/dense_71/MatMul/ReadVariableOp;autoencoder_35/sequential_71/dense_71/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
ó
Ï
1__inference_autoencoder_35_layer_call_fn_16620016
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
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_166198372
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
­
«
F__inference_dense_70_layer_call_and_return_conditional_losses_16620473

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_70/kernel/Regularizer/Square/ReadVariableOp
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
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
î"

K__inference_sequential_70_layer_call_and_return_conditional_losses_16619613

inputs#
dense_70_16619592:^ 
dense_70_16619594: 
identity

identity_1¢ dense_70/StatefulPartitionedCall¢1dense_70/kernel/Regularizer/Square/ReadVariableOp
 dense_70/StatefulPartitionedCallStatefulPartitionedCallinputsdense_70_16619592dense_70_16619594*
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
F__inference_dense_70_layer_call_and_return_conditional_losses_166195252"
 dense_70/StatefulPartitionedCallü
,dense_70/ActivityRegularizer/PartitionedCallPartitionedCall)dense_70/StatefulPartitionedCall:output:0*
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
2__inference_dense_70_activity_regularizer_166195012.
,dense_70/ActivityRegularizer/PartitionedCall¡
"dense_70/ActivityRegularizer/ShapeShape)dense_70/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_70/ActivityRegularizer/Shape®
0dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_70/ActivityRegularizer/strided_slice/stack²
2dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_1²
2dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_2
*dense_70/ActivityRegularizer/strided_sliceStridedSlice+dense_70/ActivityRegularizer/Shape:output:09dense_70/ActivityRegularizer/strided_slice/stack:output:0;dense_70/ActivityRegularizer/strided_slice/stack_1:output:0;dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_70/ActivityRegularizer/strided_slice³
!dense_70/ActivityRegularizer/CastCast3dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_70/ActivityRegularizer/CastÖ
$dense_70/ActivityRegularizer/truedivRealDiv5dense_70/ActivityRegularizer/PartitionedCall:output:0%dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_70/ActivityRegularizer/truediv¸
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_70_16619592*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulÔ
IdentityIdentity)dense_70/StatefulPartitionedCall:output:0!^dense_70/StatefulPartitionedCall2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_70/ActivityRegularizer/truediv:z:0!^dense_70/StatefulPartitionedCall2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
¬

0__inference_sequential_70_layer_call_fn_16620164

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
K__inference_sequential_70_layer_call_and_return_conditional_losses_166195472
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
0__inference_sequential_70_layer_call_fn_16619555
input_36
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_36unknown	unknown_0*
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_166195472
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
input_36
Á
¥
0__inference_sequential_71_layer_call_fn_16620308
dense_71_input
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_71_inputunknown	unknown_0*
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_166197592
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
_user_specified_namedense_71_input
ò$
Î
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16619893
x(
sequential_70_16619868:^ $
sequential_70_16619870: (
sequential_71_16619874: ^$
sequential_71_16619876:^
identity

identity_1¢1dense_70/kernel/Regularizer/Square/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¢%sequential_70/StatefulPartitionedCall¢%sequential_71/StatefulPartitionedCall±
%sequential_70/StatefulPartitionedCallStatefulPartitionedCallxsequential_70_16619868sequential_70_16619870*
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_166196132'
%sequential_70/StatefulPartitionedCallÛ
%sequential_71/StatefulPartitionedCallStatefulPartitionedCall.sequential_70/StatefulPartitionedCall:output:0sequential_71_16619874sequential_71_16619876*
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_166197592'
%sequential_71/StatefulPartitionedCall½
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_70_16619868*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mul½
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_71_16619874*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulº
IdentityIdentity.sequential_71/StatefulPartitionedCall:output:02^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp&^sequential_70/StatefulPartitionedCall&^sequential_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_70/StatefulPartitionedCall:output:12^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp&^sequential_70/StatefulPartitionedCall&^sequential_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_70/StatefulPartitionedCall%sequential_70/StatefulPartitionedCall2N
%sequential_71/StatefulPartitionedCall%sequential_71/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
¬

0__inference_sequential_70_layer_call_fn_16620174

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
K__inference_sequential_70_layer_call_and_return_conditional_losses_166196132
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
ä
³
__inference_loss_fn_0_16620413L
:dense_70_kernel_regularizer_square_readvariableop_resource:^ 
identity¢1dense_70/kernel/Regularizer/Square/ReadVariableOpá
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_70_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mul
IdentityIdentity#dense_70/kernel/Regularizer/mul:z:02^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp
²A
ä
K__inference_sequential_70_layer_call_and_return_conditional_losses_16620220

inputs9
'dense_70_matmul_readvariableop_resource:^ 6
(dense_70_biasadd_readvariableop_resource: 
identity

identity_1¢dense_70/BiasAdd/ReadVariableOp¢dense_70/MatMul/ReadVariableOp¢1dense_70/kernel/Regularizer/Square/ReadVariableOp¨
dense_70/MatMul/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02 
dense_70/MatMul/ReadVariableOp
dense_70/MatMulMatMulinputs&dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_70/MatMul§
dense_70/BiasAdd/ReadVariableOpReadVariableOp(dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_70/BiasAdd/ReadVariableOp¥
dense_70/BiasAddBiasAdddense_70/MatMul:product:0'dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_70/BiasAdd|
dense_70/SigmoidSigmoiddense_70/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_70/Sigmoid¬
3dense_70/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_70/ActivityRegularizer/Mean/reduction_indicesÇ
!dense_70/ActivityRegularizer/MeanMeandense_70/Sigmoid:y:0<dense_70/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2#
!dense_70/ActivityRegularizer/Mean
&dense_70/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2(
&dense_70/ActivityRegularizer/Maximum/yÙ
$dense_70/ActivityRegularizer/MaximumMaximum*dense_70/ActivityRegularizer/Mean:output:0/dense_70/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2&
$dense_70/ActivityRegularizer/Maximum
&dense_70/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&dense_70/ActivityRegularizer/truediv/x×
$dense_70/ActivityRegularizer/truedivRealDiv/dense_70/ActivityRegularizer/truediv/x:output:0(dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2&
$dense_70/ActivityRegularizer/truediv
 dense_70/ActivityRegularizer/LogLog(dense_70/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/Log
"dense_70/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"dense_70/ActivityRegularizer/mul/xÃ
 dense_70/ActivityRegularizer/mulMul+dense_70/ActivityRegularizer/mul/x:output:0$dense_70/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/mul
"dense_70/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dense_70/ActivityRegularizer/sub/xÇ
 dense_70/ActivityRegularizer/subSub+dense_70/ActivityRegularizer/sub/x:output:0(dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/sub
(dense_70/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2*
(dense_70/ActivityRegularizer/truediv_1/xÙ
&dense_70/ActivityRegularizer/truediv_1RealDiv1dense_70/ActivityRegularizer/truediv_1/x:output:0$dense_70/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2(
&dense_70/ActivityRegularizer/truediv_1 
"dense_70/ActivityRegularizer/Log_1Log*dense_70/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2$
"dense_70/ActivityRegularizer/Log_1
$dense_70/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2&
$dense_70/ActivityRegularizer/mul_1/xË
"dense_70/ActivityRegularizer/mul_1Mul-dense_70/ActivityRegularizer/mul_1/x:output:0&dense_70/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2$
"dense_70/ActivityRegularizer/mul_1À
 dense_70/ActivityRegularizer/addAddV2$dense_70/ActivityRegularizer/mul:z:0&dense_70/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/add
"dense_70/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_70/ActivityRegularizer/Const¿
 dense_70/ActivityRegularizer/SumSum$dense_70/ActivityRegularizer/add:z:0+dense_70/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_70/ActivityRegularizer/Sum
$dense_70/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dense_70/ActivityRegularizer/mul_2/xÊ
"dense_70/ActivityRegularizer/mul_2Mul-dense_70/ActivityRegularizer/mul_2/x:output:0)dense_70/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_70/ActivityRegularizer/mul_2
"dense_70/ActivityRegularizer/ShapeShapedense_70/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_70/ActivityRegularizer/Shape®
0dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_70/ActivityRegularizer/strided_slice/stack²
2dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_1²
2dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_2
*dense_70/ActivityRegularizer/strided_sliceStridedSlice+dense_70/ActivityRegularizer/Shape:output:09dense_70/ActivityRegularizer/strided_slice/stack:output:0;dense_70/ActivityRegularizer/strided_slice/stack_1:output:0;dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_70/ActivityRegularizer/strided_slice³
!dense_70/ActivityRegularizer/CastCast3dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_70/ActivityRegularizer/CastË
&dense_70/ActivityRegularizer/truediv_2RealDiv&dense_70/ActivityRegularizer/mul_2:z:0%dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_70/ActivityRegularizer/truediv_2Î
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulß
IdentityIdentitydense_70/Sigmoid:y:0 ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityè

Identity_1Identity*dense_70/ActivityRegularizer/truediv_2:z:0 ^dense_70/BiasAdd/ReadVariableOp^dense_70/MatMul/ReadVariableOp2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2B
dense_70/BiasAdd/ReadVariableOpdense_70/BiasAdd/ReadVariableOp2@
dense_70/MatMul/ReadVariableOpdense_70/MatMul/ReadVariableOp2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
­
«
F__inference_dense_71_layer_call_and_return_conditional_losses_16619703

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp
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
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Õ
1__inference_autoencoder_35_layer_call_fn_16619919
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
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_166198932
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620376
dense_71_input9
'dense_71_matmul_readvariableop_resource: ^6
(dense_71_biasadd_readvariableop_resource:^
identity¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¨
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_71/MatMul/ReadVariableOp
dense_71/MatMulMatMuldense_71_input&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/MatMul§
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_71/BiasAdd/ReadVariableOp¥
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/BiasAdd|
dense_71/SigmoidSigmoiddense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/SigmoidÎ
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulß
IdentityIdentitydense_71/Sigmoid:y:0 ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_71_input
Êe
º
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16620089
xG
5sequential_70_dense_70_matmul_readvariableop_resource:^ D
6sequential_70_dense_70_biasadd_readvariableop_resource: G
5sequential_71_dense_71_matmul_readvariableop_resource: ^D
6sequential_71_dense_71_biasadd_readvariableop_resource:^
identity

identity_1¢1dense_70/kernel/Regularizer/Square/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¢-sequential_70/dense_70/BiasAdd/ReadVariableOp¢,sequential_70/dense_70/MatMul/ReadVariableOp¢-sequential_71/dense_71/BiasAdd/ReadVariableOp¢,sequential_71/dense_71/MatMul/ReadVariableOpÒ
,sequential_70/dense_70/MatMul/ReadVariableOpReadVariableOp5sequential_70_dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02.
,sequential_70/dense_70/MatMul/ReadVariableOp³
sequential_70/dense_70/MatMulMatMulx4sequential_70/dense_70/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_70/dense_70/MatMulÑ
-sequential_70/dense_70/BiasAdd/ReadVariableOpReadVariableOp6sequential_70_dense_70_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_70/dense_70/BiasAdd/ReadVariableOpÝ
sequential_70/dense_70/BiasAddBiasAdd'sequential_70/dense_70/MatMul:product:05sequential_70/dense_70/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_70/dense_70/BiasAdd¦
sequential_70/dense_70/SigmoidSigmoid'sequential_70/dense_70/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_70/dense_70/SigmoidÈ
Asequential_70/dense_70/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_70/dense_70/ActivityRegularizer/Mean/reduction_indicesÿ
/sequential_70/dense_70/ActivityRegularizer/MeanMean"sequential_70/dense_70/Sigmoid:y:0Jsequential_70/dense_70/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 21
/sequential_70/dense_70/ActivityRegularizer/Mean±
4sequential_70/dense_70/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.26
4sequential_70/dense_70/ActivityRegularizer/Maximum/y
2sequential_70/dense_70/ActivityRegularizer/MaximumMaximum8sequential_70/dense_70/ActivityRegularizer/Mean:output:0=sequential_70/dense_70/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 24
2sequential_70/dense_70/ActivityRegularizer/Maximum±
4sequential_70/dense_70/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<26
4sequential_70/dense_70/ActivityRegularizer/truediv/x
2sequential_70/dense_70/ActivityRegularizer/truedivRealDiv=sequential_70/dense_70/ActivityRegularizer/truediv/x:output:06sequential_70/dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 24
2sequential_70/dense_70/ActivityRegularizer/truedivÄ
.sequential_70/dense_70/ActivityRegularizer/LogLog6sequential_70/dense_70/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/Log©
0sequential_70/dense_70/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<22
0sequential_70/dense_70/ActivityRegularizer/mul/xû
.sequential_70/dense_70/ActivityRegularizer/mulMul9sequential_70/dense_70/ActivityRegularizer/mul/x:output:02sequential_70/dense_70/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/mul©
0sequential_70/dense_70/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_70/dense_70/ActivityRegularizer/sub/xÿ
.sequential_70/dense_70/ActivityRegularizer/subSub9sequential_70/dense_70/ActivityRegularizer/sub/x:output:06sequential_70/dense_70/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/subµ
6sequential_70/dense_70/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?28
6sequential_70/dense_70/ActivityRegularizer/truediv_1/x
4sequential_70/dense_70/ActivityRegularizer/truediv_1RealDiv?sequential_70/dense_70/ActivityRegularizer/truediv_1/x:output:02sequential_70/dense_70/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 26
4sequential_70/dense_70/ActivityRegularizer/truediv_1Ê
0sequential_70/dense_70/ActivityRegularizer/Log_1Log8sequential_70/dense_70/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 22
0sequential_70/dense_70/ActivityRegularizer/Log_1­
2sequential_70/dense_70/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?24
2sequential_70/dense_70/ActivityRegularizer/mul_1/x
0sequential_70/dense_70/ActivityRegularizer/mul_1Mul;sequential_70/dense_70/ActivityRegularizer/mul_1/x:output:04sequential_70/dense_70/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 22
0sequential_70/dense_70/ActivityRegularizer/mul_1ø
.sequential_70/dense_70/ActivityRegularizer/addAddV22sequential_70/dense_70/ActivityRegularizer/mul:z:04sequential_70/dense_70/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/add®
0sequential_70/dense_70/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_70/dense_70/ActivityRegularizer/Const÷
.sequential_70/dense_70/ActivityRegularizer/SumSum2sequential_70/dense_70/ActivityRegularizer/add:z:09sequential_70/dense_70/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_70/dense_70/ActivityRegularizer/Sum­
2sequential_70/dense_70/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_70/dense_70/ActivityRegularizer/mul_2/x
0sequential_70/dense_70/ActivityRegularizer/mul_2Mul;sequential_70/dense_70/ActivityRegularizer/mul_2/x:output:07sequential_70/dense_70/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_70/dense_70/ActivityRegularizer/mul_2¶
0sequential_70/dense_70/ActivityRegularizer/ShapeShape"sequential_70/dense_70/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_70/dense_70/ActivityRegularizer/ShapeÊ
>sequential_70/dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_70/dense_70/ActivityRegularizer/strided_slice/stackÎ
@sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1Î
@sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2ä
8sequential_70/dense_70/ActivityRegularizer/strided_sliceStridedSlice9sequential_70/dense_70/ActivityRegularizer/Shape:output:0Gsequential_70/dense_70/ActivityRegularizer/strided_slice/stack:output:0Isequential_70/dense_70/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_70/dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_70/dense_70/ActivityRegularizer/strided_sliceÝ
/sequential_70/dense_70/ActivityRegularizer/CastCastAsequential_70/dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_70/dense_70/ActivityRegularizer/Cast
4sequential_70/dense_70/ActivityRegularizer/truediv_2RealDiv4sequential_70/dense_70/ActivityRegularizer/mul_2:z:03sequential_70/dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_70/dense_70/ActivityRegularizer/truediv_2Ò
,sequential_71/dense_71/MatMul/ReadVariableOpReadVariableOp5sequential_71_dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02.
,sequential_71/dense_71/MatMul/ReadVariableOpÔ
sequential_71/dense_71/MatMulMatMul"sequential_70/dense_70/Sigmoid:y:04sequential_71/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
sequential_71/dense_71/MatMulÑ
-sequential_71/dense_71/BiasAdd/ReadVariableOpReadVariableOp6sequential_71_dense_71_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02/
-sequential_71/dense_71/BiasAdd/ReadVariableOpÝ
sequential_71/dense_71/BiasAddBiasAdd'sequential_71/dense_71/MatMul:product:05sequential_71/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_71/dense_71/BiasAdd¦
sequential_71/dense_71/SigmoidSigmoid'sequential_71/dense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_71/dense_71/SigmoidÜ
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_70_dense_70_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulÜ
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_71_dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mul
IdentityIdentity"sequential_71/dense_71/Sigmoid:y:02^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp.^sequential_70/dense_70/BiasAdd/ReadVariableOp-^sequential_70/dense_70/MatMul/ReadVariableOp.^sequential_71/dense_71/BiasAdd/ReadVariableOp-^sequential_71/dense_71/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity¥

Identity_1Identity8sequential_70/dense_70/ActivityRegularizer/truediv_2:z:02^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp.^sequential_70/dense_70/BiasAdd/ReadVariableOp-^sequential_70/dense_70/MatMul/ReadVariableOp.^sequential_71/dense_71/BiasAdd/ReadVariableOp-^sequential_71/dense_71/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_70/dense_70/BiasAdd/ReadVariableOp-sequential_70/dense_70/BiasAdd/ReadVariableOp2\
,sequential_70/dense_70/MatMul/ReadVariableOp,sequential_70/dense_70/MatMul/ReadVariableOp2^
-sequential_71/dense_71/BiasAdd/ReadVariableOp-sequential_71/dense_71/BiasAdd/ReadVariableOp2\
,sequential_71/dense_71/MatMul/ReadVariableOp,sequential_71/dense_71/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX


K__inference_sequential_71_layer_call_and_return_conditional_losses_16619759

inputs#
dense_71_16619747: ^
dense_71_16619749:^
identity¢ dense_71/StatefulPartitionedCall¢1dense_71/kernel/Regularizer/Square/ReadVariableOp
 dense_71/StatefulPartitionedCallStatefulPartitionedCallinputsdense_71_16619747dense_71_16619749*
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
F__inference_dense_71_layer_call_and_return_conditional_losses_166197032"
 dense_71/StatefulPartitionedCall¸
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_71_16619747*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulÔ
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0!^dense_71/StatefulPartitionedCall2^dense_71/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
%
Ô
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16619975
input_1(
sequential_70_16619950:^ $
sequential_70_16619952: (
sequential_71_16619956: ^$
sequential_71_16619958:^
identity

identity_1¢1dense_70/kernel/Regularizer/Square/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¢%sequential_70/StatefulPartitionedCall¢%sequential_71/StatefulPartitionedCall·
%sequential_70/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_70_16619950sequential_70_16619952*
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_166196132'
%sequential_70/StatefulPartitionedCallÛ
%sequential_71/StatefulPartitionedCallStatefulPartitionedCall.sequential_70/StatefulPartitionedCall:output:0sequential_71_16619956sequential_71_16619958*
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_166197592'
%sequential_71/StatefulPartitionedCall½
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_70_16619950*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mul½
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_71_16619956*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulº
IdentityIdentity.sequential_71/StatefulPartitionedCall:output:02^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp&^sequential_70/StatefulPartitionedCall&^sequential_71/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_70/StatefulPartitionedCall:output:12^dense_70/kernel/Regularizer/Square/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp&^sequential_70/StatefulPartitionedCall&^sequential_71/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_70/StatefulPartitionedCall%sequential_70/StatefulPartitionedCall2N
%sequential_71/StatefulPartitionedCall%sequential_71/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1


K__inference_sequential_71_layer_call_and_return_conditional_losses_16619716

inputs#
dense_71_16619704: ^
dense_71_16619706:^
identity¢ dense_71/StatefulPartitionedCall¢1dense_71/kernel/Regularizer/Square/ReadVariableOp
 dense_71/StatefulPartitionedCallStatefulPartitionedCallinputsdense_71_16619704dense_71_16619706*
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
F__inference_dense_71_layer_call_and_return_conditional_losses_166197032"
 dense_71/StatefulPartitionedCall¸
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_71_16619704*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulÔ
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0!^dense_71/StatefulPartitionedCall2^dense_71/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Î
Ê
&__inference_signature_wrapper_16620002
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
#__inference__wrapped_model_166194722
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


+__inference_dense_70_layer_call_fn_16620391

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
F__inference_dense_70_layer_call_and_return_conditional_losses_166195252
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
Ç
Ü
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620359
dense_71_input9
'dense_71_matmul_readvariableop_resource: ^6
(dense_71_biasadd_readvariableop_resource:^
identity¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¨
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_71/MatMul/ReadVariableOp
dense_71/MatMulMatMuldense_71_input&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/MatMul§
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_71/BiasAdd/ReadVariableOp¥
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/BiasAdd|
dense_71/SigmoidSigmoiddense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/SigmoidÎ
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulß
IdentityIdentitydense_71/Sigmoid:y:0 ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_71_input
ó
Ï
1__inference_autoencoder_35_layer_call_fn_16620030
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
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_166198932
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_16619655
input_36#
dense_70_16619634:^ 
dense_70_16619636: 
identity

identity_1¢ dense_70/StatefulPartitionedCall¢1dense_70/kernel/Regularizer/Square/ReadVariableOp
 dense_70/StatefulPartitionedCallStatefulPartitionedCallinput_36dense_70_16619634dense_70_16619636*
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
F__inference_dense_70_layer_call_and_return_conditional_losses_166195252"
 dense_70/StatefulPartitionedCallü
,dense_70/ActivityRegularizer/PartitionedCallPartitionedCall)dense_70/StatefulPartitionedCall:output:0*
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
2__inference_dense_70_activity_regularizer_166195012.
,dense_70/ActivityRegularizer/PartitionedCall¡
"dense_70/ActivityRegularizer/ShapeShape)dense_70/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_70/ActivityRegularizer/Shape®
0dense_70/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_70/ActivityRegularizer/strided_slice/stack²
2dense_70/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_1²
2dense_70/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_70/ActivityRegularizer/strided_slice/stack_2
*dense_70/ActivityRegularizer/strided_sliceStridedSlice+dense_70/ActivityRegularizer/Shape:output:09dense_70/ActivityRegularizer/strided_slice/stack:output:0;dense_70/ActivityRegularizer/strided_slice/stack_1:output:0;dense_70/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_70/ActivityRegularizer/strided_slice³
!dense_70/ActivityRegularizer/CastCast3dense_70/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_70/ActivityRegularizer/CastÖ
$dense_70/ActivityRegularizer/truedivRealDiv5dense_70/ActivityRegularizer/PartitionedCall:output:0%dense_70/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_70/ActivityRegularizer/truediv¸
1dense_70/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_70_16619634*
_output_shapes

:^ *
dtype023
1dense_70/kernel/Regularizer/Square/ReadVariableOp¶
"dense_70/kernel/Regularizer/SquareSquare9dense_70/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_70/kernel/Regularizer/Square
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_70/kernel/Regularizer/Const¾
dense_70/kernel/Regularizer/SumSum&dense_70/kernel/Regularizer/Square:y:0*dense_70/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/Sum
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_70/kernel/Regularizer/mul/xÀ
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_70/kernel/Regularizer/mulÔ
IdentityIdentity)dense_70/StatefulPartitionedCall:output:0!^dense_70/StatefulPartitionedCall2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_70/ActivityRegularizer/truediv:z:0!^dense_70/StatefulPartitionedCall2^dense_70/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2f
1dense_70/kernel/Regularizer/Square/ReadVariableOp1dense_70/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_36
¯
Ô
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620325

inputs9
'dense_71_matmul_readvariableop_resource: ^6
(dense_71_biasadd_readvariableop_resource:^
identity¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¨
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_71/MatMul/ReadVariableOp
dense_71/MatMulMatMulinputs&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/MatMul§
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_71/BiasAdd/ReadVariableOp¥
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/BiasAdd|
dense_71/SigmoidSigmoiddense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/SigmoidÎ
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulß
IdentityIdentitydense_71/Sigmoid:y:0 ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


+__inference_dense_71_layer_call_fn_16620428

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
F__inference_dense_71_layer_call_and_return_conditional_losses_166197032
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
¨
Ç
J__inference_dense_70_layer_call_and_return_all_conditional_losses_16620402

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
F__inference_dense_70_layer_call_and_return_conditional_losses_166195252
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
2__inference_dense_70_activity_regularizer_166195012
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

Õ
1__inference_autoencoder_35_layer_call_fn_16619849
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
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_166198372
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
0__inference_sequential_71_layer_call_fn_16620299

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
K__inference_sequential_71_layer_call_and_return_conditional_losses_166197592
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620342

inputs9
'dense_71_matmul_readvariableop_resource: ^6
(dense_71_biasadd_readvariableop_resource:^
identity¢dense_71/BiasAdd/ReadVariableOp¢dense_71/MatMul/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp¨
dense_71/MatMul/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_71/MatMul/ReadVariableOp
dense_71/MatMulMatMulinputs&dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/MatMul§
dense_71/BiasAdd/ReadVariableOpReadVariableOp(dense_71_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_71/BiasAdd/ReadVariableOp¥
dense_71/BiasAddBiasAdddense_71/MatMul:product:0'dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/BiasAdd|
dense_71/SigmoidSigmoiddense_71/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_71/SigmoidÎ
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_71_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulß
IdentityIdentitydense_71/Sigmoid:y:0 ^dense_71/BiasAdd/ReadVariableOp^dense_71/MatMul/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_71/BiasAdd/ReadVariableOpdense_71/BiasAdd/ReadVariableOp2@
dense_71/MatMul/ReadVariableOpdense_71/MatMul/ReadVariableOp2f
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û
æ
$__inference__traced_restore_16620530
file_prefix2
 assignvariableop_dense_70_kernel:^ .
 assignvariableop_1_dense_70_bias: 4
"assignvariableop_2_dense_71_kernel: ^.
 assignvariableop_3_dense_71_bias:^

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
AssignVariableOpAssignVariableOp assignvariableop_dense_70_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_70_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_71_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_71_biasIdentity_3:output:0"/device:CPU:0*
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
©

0__inference_sequential_71_layer_call_fn_16620290

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
K__inference_sequential_71_layer_call_and_return_conditional_losses_166197162
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
F__inference_dense_71_layer_call_and_return_conditional_losses_16620445

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_71/kernel/Regularizer/Square/ReadVariableOp
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
1dense_71/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_71/kernel/Regularizer/Square/ReadVariableOp¶
"dense_71/kernel/Regularizer/SquareSquare9dense_71/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_71/kernel/Regularizer/Square
!dense_71/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_71/kernel/Regularizer/Const¾
dense_71/kernel/Regularizer/SumSum&dense_71/kernel/Regularizer/Square:y:0*dense_71/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/Sum
!dense_71/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_71/kernel/Regularizer/mul/xÀ
dense_71/kernel/Regularizer/mulMul*dense_71/kernel/Regularizer/mul/x:output:0(dense_71/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_71/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_71/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_71/kernel/Regularizer/Square/ReadVariableOp1dense_71/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
³
R
2__inference_dense_70_activity_regularizer_16619501

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
Ç
ª
!__inference__traced_save_16620508
file_prefix.
*savev2_dense_70_kernel_read_readvariableop,
(savev2_dense_70_bias_read_readvariableop.
*savev2_dense_71_kernel_read_readvariableop,
(savev2_dense_71_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_70_kernel_read_readvariableop(savev2_dense_70_bias_read_readvariableop*savev2_dense_71_kernel_read_readvariableop(savev2_dense_71_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
: "ÌL
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
_tf_keras_model{"name": "autoencoder_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
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
_tf_keras_sequentialÞ{"name": "sequential_70", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_70", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_36"}}, {"class_name": "Dense", "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_36"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_70", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_36"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
¾
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_sequentialé{"name": "sequential_71", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_71", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_71_input"}}, {"class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_71_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_71", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_71_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
{"name": "dense_70", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_70", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
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
_tf_keras_layer®{"name": "dense_71", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_71", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
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
!:^ 2dense_70/kernel
: 2dense_70/bias
!: ^2dense_71/kernel
:^2dense_71/bias
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
1__inference_autoencoder_35_layer_call_fn_16619849
1__inference_autoencoder_35_layer_call_fn_16620016
1__inference_autoencoder_35_layer_call_fn_16620030
1__inference_autoencoder_35_layer_call_fn_16619919®
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
#__inference__wrapped_model_16619472¶
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
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16620089
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16620148
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16619947
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16619975®
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
0__inference_sequential_70_layer_call_fn_16619555
0__inference_sequential_70_layer_call_fn_16620164
0__inference_sequential_70_layer_call_fn_16620174
0__inference_sequential_70_layer_call_fn_16619631À
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_16620220
K__inference_sequential_70_layer_call_and_return_conditional_losses_16620266
K__inference_sequential_70_layer_call_and_return_conditional_losses_16619655
K__inference_sequential_70_layer_call_and_return_conditional_losses_16619679À
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
0__inference_sequential_71_layer_call_fn_16620281
0__inference_sequential_71_layer_call_fn_16620290
0__inference_sequential_71_layer_call_fn_16620299
0__inference_sequential_71_layer_call_fn_16620308À
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620325
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620342
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620359
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620376À
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
&__inference_signature_wrapper_16620002input_1"
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
+__inference_dense_70_layer_call_fn_16620391¢
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
J__inference_dense_70_layer_call_and_return_all_conditional_losses_16620402¢
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
__inference_loss_fn_0_16620413
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
+__inference_dense_71_layer_call_fn_16620428¢
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
F__inference_dense_71_layer_call_and_return_conditional_losses_16620445¢
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
__inference_loss_fn_1_16620456
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
2__inference_dense_70_activity_regularizer_16619501²
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
F__inference_dense_70_layer_call_and_return_conditional_losses_16620473¢
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
#__inference__wrapped_model_16619472m0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ^
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ^Á
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16619947q4¢1
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
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16619975q4¢1
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
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16620089k.¢+
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
L__inference_autoencoder_35_layer_call_and_return_conditional_losses_16620148k.¢+
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
1__inference_autoencoder_35_layer_call_fn_16619849V4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p 
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_35_layer_call_fn_16619919V4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_35_layer_call_fn_16620016P.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p 
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_35_layer_call_fn_16620030P.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p
ª "ÿÿÿÿÿÿÿÿÿ^e
2__inference_dense_70_activity_regularizer_16619501/$¢!
¢


activation
ª " ¸
J__inference_dense_70_layer_call_and_return_all_conditional_losses_16620402j/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 ¦
F__inference_dense_70_layer_call_and_return_conditional_losses_16620473\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_dense_70_layer_call_fn_16620391O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "ÿÿÿÿÿÿÿÿÿ ¦
F__inference_dense_71_layer_call_and_return_conditional_losses_16620445\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 ~
+__inference_dense_71_layer_call_fn_16620428O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ^=
__inference_loss_fn_0_16620413¢

¢ 
ª " =
__inference_loss_fn_1_16620456¢

¢ 
ª " Ã
K__inference_sequential_70_layer_call_and_return_conditional_losses_16619655t9¢6
/¢,
"
input_36ÿÿÿÿÿÿÿÿÿ^
p 

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Ã
K__inference_sequential_70_layer_call_and_return_conditional_losses_16619679t9¢6
/¢,
"
input_36ÿÿÿÿÿÿÿÿÿ^
p

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Á
K__inference_sequential_70_layer_call_and_return_conditional_losses_16620220r7¢4
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
K__inference_sequential_70_layer_call_and_return_conditional_losses_16620266r7¢4
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
0__inference_sequential_70_layer_call_fn_16619555Y9¢6
/¢,
"
input_36ÿÿÿÿÿÿÿÿÿ^
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_70_layer_call_fn_16619631Y9¢6
/¢,
"
input_36ÿÿÿÿÿÿÿÿÿ^
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_70_layer_call_fn_16620164W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_70_layer_call_fn_16620174W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p

 
ª "ÿÿÿÿÿÿÿÿÿ ³
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620325d7¢4
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620342d7¢4
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
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620359l?¢<
5¢2
(%
dense_71_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 »
K__inference_sequential_71_layer_call_and_return_conditional_losses_16620376l?¢<
5¢2
(%
dense_71_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 
0__inference_sequential_71_layer_call_fn_16620281_?¢<
5¢2
(%
dense_71_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_71_layer_call_fn_16620290W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_71_layer_call_fn_16620299W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_71_layer_call_fn_16620308_?¢<
5¢2
(%
dense_71_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ^¢
&__inference_signature_wrapper_16620002x;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ^"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ^