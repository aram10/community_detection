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
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ * 
shared_namedense_40/kernel
s
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes

:^ *
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes
: *
dtype0
z
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^* 
shared_namedense_41/kernel
s
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes

: ^*
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
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
VARIABLE_VALUEdense_40/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_40/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_41/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_41/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_40/kerneldense_40/biasdense_41/kerneldense_41/bias*
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
&__inference_signature_wrapper_16601237
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
±
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_16601743
Ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_40/kerneldense_40/biasdense_41/kerneldense_41/bias*
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
$__inference__traced_restore_16601765¥ô
ô"

K__inference_sequential_40_layer_call_and_return_conditional_losses_16600890
input_21#
dense_40_16600869:^ 
dense_40_16600871: 
identity

identity_1¢ dense_40/StatefulPartitionedCall¢1dense_40/kernel/Regularizer/Square/ReadVariableOp
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinput_21dense_40_16600869dense_40_16600871*
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
F__inference_dense_40_layer_call_and_return_conditional_losses_166007602"
 dense_40/StatefulPartitionedCallü
,dense_40/ActivityRegularizer/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
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
2__inference_dense_40_activity_regularizer_166007362.
,dense_40/ActivityRegularizer/PartitionedCall¡
"dense_40/ActivityRegularizer/ShapeShape)dense_40/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_40/ActivityRegularizer/Shape®
0dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_40/ActivityRegularizer/strided_slice/stack²
2dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_1²
2dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_2
*dense_40/ActivityRegularizer/strided_sliceStridedSlice+dense_40/ActivityRegularizer/Shape:output:09dense_40/ActivityRegularizer/strided_slice/stack:output:0;dense_40/ActivityRegularizer/strided_slice/stack_1:output:0;dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_40/ActivityRegularizer/strided_slice³
!dense_40/ActivityRegularizer/CastCast3dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_40/ActivityRegularizer/CastÖ
$dense_40/ActivityRegularizer/truedivRealDiv5dense_40/ActivityRegularizer/PartitionedCall:output:0%dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_40/ActivityRegularizer/truediv¸
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_16600869*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulÔ
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_40/ActivityRegularizer/truediv:z:0!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_21
²

0__inference_sequential_40_layer_call_fn_16600790
input_21
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0*
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_166007822
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
input_21
ä
³
__inference_loss_fn_1_16601691L
:dense_41_kernel_regularizer_square_readvariableop_resource: ^
identity¢1dense_41/kernel/Regularizer/Square/ReadVariableOpá
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_41_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul
IdentityIdentity#dense_41/kernel/Regularizer/mul:z:02^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp
ò$
Î
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601072
x(
sequential_40_16601047:^ $
sequential_40_16601049: (
sequential_41_16601053: ^$
sequential_41_16601055:^
identity

identity_1¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢%sequential_40/StatefulPartitionedCall¢%sequential_41/StatefulPartitionedCall±
%sequential_40/StatefulPartitionedCallStatefulPartitionedCallxsequential_40_16601047sequential_40_16601049*
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_166007822'
%sequential_40/StatefulPartitionedCallÛ
%sequential_41/StatefulPartitionedCallStatefulPartitionedCall.sequential_40/StatefulPartitionedCall:output:0sequential_41_16601053sequential_41_16601055*
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_166009512'
%sequential_41/StatefulPartitionedCall½
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_40_16601047*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul½
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_41_16601053*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulº
IdentityIdentity.sequential_41/StatefulPartitionedCall:output:02^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp&^sequential_40/StatefulPartitionedCall&^sequential_41/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_40/StatefulPartitionedCall:output:12^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp&^sequential_40/StatefulPartitionedCall&^sequential_41/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_40/StatefulPartitionedCall%sequential_40/StatefulPartitionedCall2N
%sequential_41/StatefulPartitionedCall%sequential_41/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
¨
Ç
J__inference_dense_40_layer_call_and_return_all_conditional_losses_16601637

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
F__inference_dense_40_layer_call_and_return_conditional_losses_166007602
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
2__inference_dense_40_activity_regularizer_166007362
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
­
«
F__inference_dense_40_layer_call_and_return_conditional_losses_16601708

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_40/kernel/Regularizer/Square/ReadVariableOp
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
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs


+__inference_dense_40_layer_call_fn_16601626

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
F__inference_dense_40_layer_call_and_return_conditional_losses_166007602
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
­
«
F__inference_dense_41_layer_call_and_return_conditional_losses_16600938

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp
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
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ä]

#__inference__wrapped_model_16600707
input_1V
Dautoencoder_20_sequential_40_dense_40_matmul_readvariableop_resource:^ S
Eautoencoder_20_sequential_40_dense_40_biasadd_readvariableop_resource: V
Dautoencoder_20_sequential_41_dense_41_matmul_readvariableop_resource: ^S
Eautoencoder_20_sequential_41_dense_41_biasadd_readvariableop_resource:^
identity¢<autoencoder_20/sequential_40/dense_40/BiasAdd/ReadVariableOp¢;autoencoder_20/sequential_40/dense_40/MatMul/ReadVariableOp¢<autoencoder_20/sequential_41/dense_41/BiasAdd/ReadVariableOp¢;autoencoder_20/sequential_41/dense_41/MatMul/ReadVariableOpÿ
;autoencoder_20/sequential_40/dense_40/MatMul/ReadVariableOpReadVariableOpDautoencoder_20_sequential_40_dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02=
;autoencoder_20/sequential_40/dense_40/MatMul/ReadVariableOpæ
,autoencoder_20/sequential_40/dense_40/MatMulMatMulinput_1Cautoencoder_20/sequential_40/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,autoencoder_20/sequential_40/dense_40/MatMulþ
<autoencoder_20/sequential_40/dense_40/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_20_sequential_40_dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<autoencoder_20/sequential_40/dense_40/BiasAdd/ReadVariableOp
-autoencoder_20/sequential_40/dense_40/BiasAddBiasAdd6autoencoder_20/sequential_40/dense_40/MatMul:product:0Dautoencoder_20/sequential_40/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-autoencoder_20/sequential_40/dense_40/BiasAddÓ
-autoencoder_20/sequential_40/dense_40/SigmoidSigmoid6autoencoder_20/sequential_40/dense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2/
-autoencoder_20/sequential_40/dense_40/Sigmoidæ
Pautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2R
Pautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Mean/reduction_indices»
>autoencoder_20/sequential_40/dense_40/ActivityRegularizer/MeanMean1autoencoder_20/sequential_40/dense_40/Sigmoid:y:0Yautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2@
>autoencoder_20/sequential_40/dense_40/ActivityRegularizer/MeanÏ
Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2E
Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Maximum/yÍ
Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/MaximumMaximumGautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Mean:output:0Lautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/MaximumÏ
Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2E
Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv/xË
Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truedivRealDivLautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv/x:output:0Eautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truedivñ
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/LogLogEautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2?
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/LogÇ
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2A
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul/x·
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/mulMulHautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul/x:output:0Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2?
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/mulÇ
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/sub/x»
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/subSubHautoencoder_20/sequential_40/dense_40/ActivityRegularizer/sub/x:output:0Eautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2?
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/subÓ
Eautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2G
Eautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv_1/xÍ
Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv_1RealDivNautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv_1/x:output:0Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv_1÷
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/Log_1LogGautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/Log_1Ë
Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2C
Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_1/x¿
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_1MulJautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_1/x:output:0Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2A
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_1´
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/addAddV2Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul:z:0Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2?
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/addÌ
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2A
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/Const³
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/SumSumAautoencoder_20/sequential_40/dense_40/ActivityRegularizer/add:z:0Hautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2?
=autoencoder_20/sequential_40/dense_40/ActivityRegularizer/SumË
Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2C
Aautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_2/x¾
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_2MulJautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_2/x:output:0Fautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_2ã
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/ShapeShape1autoencoder_20/sequential_40/dense_40/Sigmoid:y:0*
T0*
_output_shapes
:2A
?autoencoder_20/sequential_40/dense_40/ActivityRegularizer/Shapeè
Mautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Mautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stackì
Oautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1ì
Oautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Oautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2¾
Gautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_sliceStridedSliceHautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Shape:output:0Vautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stack:output:0Xautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1:output:0Xautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice
>autoencoder_20/sequential_40/dense_40/ActivityRegularizer/CastCastPautoencoder_20/sequential_40/dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2@
>autoencoder_20/sequential_40/dense_40/ActivityRegularizer/Cast¿
Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv_2RealDivCautoencoder_20/sequential_40/dense_40/ActivityRegularizer/mul_2:z:0Bautoencoder_20/sequential_40/dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2E
Cautoencoder_20/sequential_40/dense_40/ActivityRegularizer/truediv_2ÿ
;autoencoder_20/sequential_41/dense_41/MatMul/ReadVariableOpReadVariableOpDautoencoder_20_sequential_41_dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02=
;autoencoder_20/sequential_41/dense_41/MatMul/ReadVariableOp
,autoencoder_20/sequential_41/dense_41/MatMulMatMul1autoencoder_20/sequential_40/dense_40/Sigmoid:y:0Cautoencoder_20/sequential_41/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2.
,autoencoder_20/sequential_41/dense_41/MatMulþ
<autoencoder_20/sequential_41/dense_41/BiasAdd/ReadVariableOpReadVariableOpEautoencoder_20_sequential_41_dense_41_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02>
<autoencoder_20/sequential_41/dense_41/BiasAdd/ReadVariableOp
-autoencoder_20/sequential_41/dense_41/BiasAddBiasAdd6autoencoder_20/sequential_41/dense_41/MatMul:product:0Dautoencoder_20/sequential_41/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2/
-autoencoder_20/sequential_41/dense_41/BiasAddÓ
-autoencoder_20/sequential_41/dense_41/SigmoidSigmoid6autoencoder_20/sequential_41/dense_41/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2/
-autoencoder_20/sequential_41/dense_41/Sigmoidÿ
IdentityIdentity1autoencoder_20/sequential_41/dense_41/Sigmoid:y:0=^autoencoder_20/sequential_40/dense_40/BiasAdd/ReadVariableOp<^autoencoder_20/sequential_40/dense_40/MatMul/ReadVariableOp=^autoencoder_20/sequential_41/dense_41/BiasAdd/ReadVariableOp<^autoencoder_20/sequential_41/dense_41/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2|
<autoencoder_20/sequential_40/dense_40/BiasAdd/ReadVariableOp<autoencoder_20/sequential_40/dense_40/BiasAdd/ReadVariableOp2z
;autoencoder_20/sequential_40/dense_40/MatMul/ReadVariableOp;autoencoder_20/sequential_40/dense_40/MatMul/ReadVariableOp2|
<autoencoder_20/sequential_41/dense_41/BiasAdd/ReadVariableOp<autoencoder_20/sequential_41/dense_41/BiasAdd/ReadVariableOp2z
;autoencoder_20/sequential_41/dense_41/MatMul/ReadVariableOp;autoencoder_20/sequential_41/dense_41/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
%
Ô
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601210
input_1(
sequential_40_16601185:^ $
sequential_40_16601187: (
sequential_41_16601191: ^$
sequential_41_16601193:^
identity

identity_1¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢%sequential_40/StatefulPartitionedCall¢%sequential_41/StatefulPartitionedCall·
%sequential_40/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_40_16601185sequential_40_16601187*
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_166008482'
%sequential_40/StatefulPartitionedCallÛ
%sequential_41/StatefulPartitionedCallStatefulPartitionedCall.sequential_40/StatefulPartitionedCall:output:0sequential_41_16601191sequential_41_16601193*
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_166009942'
%sequential_41/StatefulPartitionedCall½
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_40_16601185*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul½
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_41_16601191*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulº
IdentityIdentity.sequential_41/StatefulPartitionedCall:output:02^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp&^sequential_40/StatefulPartitionedCall&^sequential_41/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_40/StatefulPartitionedCall:output:12^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp&^sequential_40/StatefulPartitionedCall&^sequential_41/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_40/StatefulPartitionedCall%sequential_40/StatefulPartitionedCall2N
%sequential_41/StatefulPartitionedCall%sequential_41/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
©

0__inference_sequential_41_layer_call_fn_16601534

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
K__inference_sequential_41_layer_call_and_return_conditional_losses_166009942
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
Á
¥
0__inference_sequential_41_layer_call_fn_16601543
dense_41_input
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_41_inputunknown	unknown_0*
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_166009942
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
_user_specified_namedense_41_input
¯
Ô
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601577

inputs9
'dense_41_matmul_readvariableop_resource: ^6
(dense_41_biasadd_readvariableop_resource:^
identity¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¨
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_41/MatMul/ReadVariableOp
dense_41/MatMulMatMulinputs&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/MatMul§
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_41/BiasAdd/ReadVariableOp¥
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/BiasAdd|
dense_41/SigmoidSigmoiddense_41/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/SigmoidÎ
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulß
IdentityIdentitydense_41/Sigmoid:y:0 ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­
«
F__inference_dense_40_layer_call_and_return_conditional_losses_16600760

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_40/kernel/Regularizer/Square/ReadVariableOp
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
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
³
R
2__inference_dense_40_activity_regularizer_16600736

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
¬

0__inference_sequential_40_layer_call_fn_16601399

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
K__inference_sequential_40_layer_call_and_return_conditional_losses_166007822
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
Î
Ê
&__inference_signature_wrapper_16601237
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
#__inference__wrapped_model_166007072
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
1__inference_autoencoder_20_layer_call_fn_16601154
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_166011282
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601594
dense_41_input9
'dense_41_matmul_readvariableop_resource: ^6
(dense_41_biasadd_readvariableop_resource:^
identity¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¨
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_41/MatMul/ReadVariableOp
dense_41/MatMulMatMuldense_41_input&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/MatMul§
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_41/BiasAdd/ReadVariableOp¥
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/BiasAdd|
dense_41/SigmoidSigmoiddense_41/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/SigmoidÎ
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulß
IdentityIdentitydense_41/Sigmoid:y:0 ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_41_input
²A
ä
K__inference_sequential_40_layer_call_and_return_conditional_losses_16601501

inputs9
'dense_40_matmul_readvariableop_resource:^ 6
(dense_40_biasadd_readvariableop_resource: 
identity

identity_1¢dense_40/BiasAdd/ReadVariableOp¢dense_40/MatMul/ReadVariableOp¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¨
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02 
dense_40/MatMul/ReadVariableOp
dense_40/MatMulMatMulinputs&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_40/MatMul§
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_40/BiasAdd/ReadVariableOp¥
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_40/BiasAdd|
dense_40/SigmoidSigmoiddense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_40/Sigmoid¬
3dense_40/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_40/ActivityRegularizer/Mean/reduction_indicesÇ
!dense_40/ActivityRegularizer/MeanMeandense_40/Sigmoid:y:0<dense_40/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2#
!dense_40/ActivityRegularizer/Mean
&dense_40/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2(
&dense_40/ActivityRegularizer/Maximum/yÙ
$dense_40/ActivityRegularizer/MaximumMaximum*dense_40/ActivityRegularizer/Mean:output:0/dense_40/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2&
$dense_40/ActivityRegularizer/Maximum
&dense_40/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&dense_40/ActivityRegularizer/truediv/x×
$dense_40/ActivityRegularizer/truedivRealDiv/dense_40/ActivityRegularizer/truediv/x:output:0(dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2&
$dense_40/ActivityRegularizer/truediv
 dense_40/ActivityRegularizer/LogLog(dense_40/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/Log
"dense_40/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"dense_40/ActivityRegularizer/mul/xÃ
 dense_40/ActivityRegularizer/mulMul+dense_40/ActivityRegularizer/mul/x:output:0$dense_40/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/mul
"dense_40/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dense_40/ActivityRegularizer/sub/xÇ
 dense_40/ActivityRegularizer/subSub+dense_40/ActivityRegularizer/sub/x:output:0(dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/sub
(dense_40/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2*
(dense_40/ActivityRegularizer/truediv_1/xÙ
&dense_40/ActivityRegularizer/truediv_1RealDiv1dense_40/ActivityRegularizer/truediv_1/x:output:0$dense_40/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2(
&dense_40/ActivityRegularizer/truediv_1 
"dense_40/ActivityRegularizer/Log_1Log*dense_40/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2$
"dense_40/ActivityRegularizer/Log_1
$dense_40/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2&
$dense_40/ActivityRegularizer/mul_1/xË
"dense_40/ActivityRegularizer/mul_1Mul-dense_40/ActivityRegularizer/mul_1/x:output:0&dense_40/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2$
"dense_40/ActivityRegularizer/mul_1À
 dense_40/ActivityRegularizer/addAddV2$dense_40/ActivityRegularizer/mul:z:0&dense_40/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/add
"dense_40/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_40/ActivityRegularizer/Const¿
 dense_40/ActivityRegularizer/SumSum$dense_40/ActivityRegularizer/add:z:0+dense_40/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/Sum
$dense_40/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dense_40/ActivityRegularizer/mul_2/xÊ
"dense_40/ActivityRegularizer/mul_2Mul-dense_40/ActivityRegularizer/mul_2/x:output:0)dense_40/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_40/ActivityRegularizer/mul_2
"dense_40/ActivityRegularizer/ShapeShapedense_40/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_40/ActivityRegularizer/Shape®
0dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_40/ActivityRegularizer/strided_slice/stack²
2dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_1²
2dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_2
*dense_40/ActivityRegularizer/strided_sliceStridedSlice+dense_40/ActivityRegularizer/Shape:output:09dense_40/ActivityRegularizer/strided_slice/stack:output:0;dense_40/ActivityRegularizer/strided_slice/stack_1:output:0;dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_40/ActivityRegularizer/strided_slice³
!dense_40/ActivityRegularizer/CastCast3dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_40/ActivityRegularizer/CastË
&dense_40/ActivityRegularizer/truediv_2RealDiv&dense_40/ActivityRegularizer/mul_2:z:0%dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_40/ActivityRegularizer/truediv_2Î
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulß
IdentityIdentitydense_40/Sigmoid:y:0 ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityè

Identity_1Identity*dense_40/ActivityRegularizer/truediv_2:z:0 ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
ó
Ï
1__inference_autoencoder_20_layer_call_fn_16601251
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_166010722
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601383
xG
5sequential_40_dense_40_matmul_readvariableop_resource:^ D
6sequential_40_dense_40_biasadd_readvariableop_resource: G
5sequential_41_dense_41_matmul_readvariableop_resource: ^D
6sequential_41_dense_41_biasadd_readvariableop_resource:^
identity

identity_1¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢-sequential_40/dense_40/BiasAdd/ReadVariableOp¢,sequential_40/dense_40/MatMul/ReadVariableOp¢-sequential_41/dense_41/BiasAdd/ReadVariableOp¢,sequential_41/dense_41/MatMul/ReadVariableOpÒ
,sequential_40/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_40_dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02.
,sequential_40/dense_40/MatMul/ReadVariableOp³
sequential_40/dense_40/MatMulMatMulx4sequential_40/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_40/dense_40/MatMulÑ
-sequential_40/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_40_dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_40/dense_40/BiasAdd/ReadVariableOpÝ
sequential_40/dense_40/BiasAddBiasAdd'sequential_40/dense_40/MatMul:product:05sequential_40/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_40/dense_40/BiasAdd¦
sequential_40/dense_40/SigmoidSigmoid'sequential_40/dense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_40/dense_40/SigmoidÈ
Asequential_40/dense_40/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_40/dense_40/ActivityRegularizer/Mean/reduction_indicesÿ
/sequential_40/dense_40/ActivityRegularizer/MeanMean"sequential_40/dense_40/Sigmoid:y:0Jsequential_40/dense_40/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 21
/sequential_40/dense_40/ActivityRegularizer/Mean±
4sequential_40/dense_40/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.26
4sequential_40/dense_40/ActivityRegularizer/Maximum/y
2sequential_40/dense_40/ActivityRegularizer/MaximumMaximum8sequential_40/dense_40/ActivityRegularizer/Mean:output:0=sequential_40/dense_40/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 24
2sequential_40/dense_40/ActivityRegularizer/Maximum±
4sequential_40/dense_40/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<26
4sequential_40/dense_40/ActivityRegularizer/truediv/x
2sequential_40/dense_40/ActivityRegularizer/truedivRealDiv=sequential_40/dense_40/ActivityRegularizer/truediv/x:output:06sequential_40/dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 24
2sequential_40/dense_40/ActivityRegularizer/truedivÄ
.sequential_40/dense_40/ActivityRegularizer/LogLog6sequential_40/dense_40/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/Log©
0sequential_40/dense_40/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<22
0sequential_40/dense_40/ActivityRegularizer/mul/xû
.sequential_40/dense_40/ActivityRegularizer/mulMul9sequential_40/dense_40/ActivityRegularizer/mul/x:output:02sequential_40/dense_40/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/mul©
0sequential_40/dense_40/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_40/dense_40/ActivityRegularizer/sub/xÿ
.sequential_40/dense_40/ActivityRegularizer/subSub9sequential_40/dense_40/ActivityRegularizer/sub/x:output:06sequential_40/dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/subµ
6sequential_40/dense_40/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?28
6sequential_40/dense_40/ActivityRegularizer/truediv_1/x
4sequential_40/dense_40/ActivityRegularizer/truediv_1RealDiv?sequential_40/dense_40/ActivityRegularizer/truediv_1/x:output:02sequential_40/dense_40/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 26
4sequential_40/dense_40/ActivityRegularizer/truediv_1Ê
0sequential_40/dense_40/ActivityRegularizer/Log_1Log8sequential_40/dense_40/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 22
0sequential_40/dense_40/ActivityRegularizer/Log_1­
2sequential_40/dense_40/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?24
2sequential_40/dense_40/ActivityRegularizer/mul_1/x
0sequential_40/dense_40/ActivityRegularizer/mul_1Mul;sequential_40/dense_40/ActivityRegularizer/mul_1/x:output:04sequential_40/dense_40/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 22
0sequential_40/dense_40/ActivityRegularizer/mul_1ø
.sequential_40/dense_40/ActivityRegularizer/addAddV22sequential_40/dense_40/ActivityRegularizer/mul:z:04sequential_40/dense_40/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/add®
0sequential_40/dense_40/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_40/dense_40/ActivityRegularizer/Const÷
.sequential_40/dense_40/ActivityRegularizer/SumSum2sequential_40/dense_40/ActivityRegularizer/add:z:09sequential_40/dense_40/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/Sum­
2sequential_40/dense_40/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_40/dense_40/ActivityRegularizer/mul_2/x
0sequential_40/dense_40/ActivityRegularizer/mul_2Mul;sequential_40/dense_40/ActivityRegularizer/mul_2/x:output:07sequential_40/dense_40/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_40/dense_40/ActivityRegularizer/mul_2¶
0sequential_40/dense_40/ActivityRegularizer/ShapeShape"sequential_40/dense_40/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_40/dense_40/ActivityRegularizer/ShapeÊ
>sequential_40/dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_40/dense_40/ActivityRegularizer/strided_slice/stackÎ
@sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1Î
@sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2ä
8sequential_40/dense_40/ActivityRegularizer/strided_sliceStridedSlice9sequential_40/dense_40/ActivityRegularizer/Shape:output:0Gsequential_40/dense_40/ActivityRegularizer/strided_slice/stack:output:0Isequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_40/dense_40/ActivityRegularizer/strided_sliceÝ
/sequential_40/dense_40/ActivityRegularizer/CastCastAsequential_40/dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_40/dense_40/ActivityRegularizer/Cast
4sequential_40/dense_40/ActivityRegularizer/truediv_2RealDiv4sequential_40/dense_40/ActivityRegularizer/mul_2:z:03sequential_40/dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_40/dense_40/ActivityRegularizer/truediv_2Ò
,sequential_41/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_41_dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02.
,sequential_41/dense_41/MatMul/ReadVariableOpÔ
sequential_41/dense_41/MatMulMatMul"sequential_40/dense_40/Sigmoid:y:04sequential_41/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
sequential_41/dense_41/MatMulÑ
-sequential_41/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_41_dense_41_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02/
-sequential_41/dense_41/BiasAdd/ReadVariableOpÝ
sequential_41/dense_41/BiasAddBiasAdd'sequential_41/dense_41/MatMul:product:05sequential_41/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_41/dense_41/BiasAdd¦
sequential_41/dense_41/SigmoidSigmoid'sequential_41/dense_41/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_41/dense_41/SigmoidÜ
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_40_dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulÜ
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_41_dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul
IdentityIdentity"sequential_41/dense_41/Sigmoid:y:02^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp.^sequential_40/dense_40/BiasAdd/ReadVariableOp-^sequential_40/dense_40/MatMul/ReadVariableOp.^sequential_41/dense_41/BiasAdd/ReadVariableOp-^sequential_41/dense_41/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity¥

Identity_1Identity8sequential_40/dense_40/ActivityRegularizer/truediv_2:z:02^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp.^sequential_40/dense_40/BiasAdd/ReadVariableOp-^sequential_40/dense_40/MatMul/ReadVariableOp.^sequential_41/dense_41/BiasAdd/ReadVariableOp-^sequential_41/dense_41/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_40/dense_40/BiasAdd/ReadVariableOp-sequential_40/dense_40/BiasAdd/ReadVariableOp2\
,sequential_40/dense_40/MatMul/ReadVariableOp,sequential_40/dense_40/MatMul/ReadVariableOp2^
-sequential_41/dense_41/BiasAdd/ReadVariableOp-sequential_41/dense_41/BiasAdd/ReadVariableOp2\
,sequential_41/dense_41/MatMul/ReadVariableOp,sequential_41/dense_41/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX


K__inference_sequential_41_layer_call_and_return_conditional_losses_16600951

inputs#
dense_41_16600939: ^
dense_41_16600941:^
identity¢ dense_41/StatefulPartitionedCall¢1dense_41/kernel/Regularizer/Square/ReadVariableOp
 dense_41/StatefulPartitionedCallStatefulPartitionedCallinputsdense_41_16600939dense_41_16600941*
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
F__inference_dense_41_layer_call_and_return_conditional_losses_166009382"
 dense_41/StatefulPartitionedCall¸
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_41_16600939*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulÔ
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_41/StatefulPartitionedCall2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Á
¥
0__inference_sequential_41_layer_call_fn_16601516
dense_41_input
unknown: ^
	unknown_0:^
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_41_inputunknown	unknown_0*
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_166009512
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
_user_specified_namedense_41_input
Û
æ
$__inference__traced_restore_16601765
file_prefix2
 assignvariableop_dense_40_kernel:^ .
 assignvariableop_1_dense_40_bias: 4
"assignvariableop_2_dense_41_kernel: ^.
 assignvariableop_3_dense_41_bias:^

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
AssignVariableOpAssignVariableOp assignvariableop_dense_40_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_40_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_41_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_41_biasIdentity_3:output:0"/device:CPU:0*
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
²A
ä
K__inference_sequential_40_layer_call_and_return_conditional_losses_16601455

inputs9
'dense_40_matmul_readvariableop_resource:^ 6
(dense_40_biasadd_readvariableop_resource: 
identity

identity_1¢dense_40/BiasAdd/ReadVariableOp¢dense_40/MatMul/ReadVariableOp¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¨
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02 
dense_40/MatMul/ReadVariableOp
dense_40/MatMulMatMulinputs&dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_40/MatMul§
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_40/BiasAdd/ReadVariableOp¥
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_40/BiasAdd|
dense_40/SigmoidSigmoiddense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_40/Sigmoid¬
3dense_40/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_40/ActivityRegularizer/Mean/reduction_indicesÇ
!dense_40/ActivityRegularizer/MeanMeandense_40/Sigmoid:y:0<dense_40/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2#
!dense_40/ActivityRegularizer/Mean
&dense_40/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.2(
&dense_40/ActivityRegularizer/Maximum/yÙ
$dense_40/ActivityRegularizer/MaximumMaximum*dense_40/ActivityRegularizer/Mean:output:0/dense_40/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2&
$dense_40/ActivityRegularizer/Maximum
&dense_40/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2(
&dense_40/ActivityRegularizer/truediv/x×
$dense_40/ActivityRegularizer/truedivRealDiv/dense_40/ActivityRegularizer/truediv/x:output:0(dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2&
$dense_40/ActivityRegularizer/truediv
 dense_40/ActivityRegularizer/LogLog(dense_40/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/Log
"dense_40/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2$
"dense_40/ActivityRegularizer/mul/xÃ
 dense_40/ActivityRegularizer/mulMul+dense_40/ActivityRegularizer/mul/x:output:0$dense_40/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/mul
"dense_40/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"dense_40/ActivityRegularizer/sub/xÇ
 dense_40/ActivityRegularizer/subSub+dense_40/ActivityRegularizer/sub/x:output:0(dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/sub
(dense_40/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2*
(dense_40/ActivityRegularizer/truediv_1/xÙ
&dense_40/ActivityRegularizer/truediv_1RealDiv1dense_40/ActivityRegularizer/truediv_1/x:output:0$dense_40/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2(
&dense_40/ActivityRegularizer/truediv_1 
"dense_40/ActivityRegularizer/Log_1Log*dense_40/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2$
"dense_40/ActivityRegularizer/Log_1
$dense_40/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?2&
$dense_40/ActivityRegularizer/mul_1/xË
"dense_40/ActivityRegularizer/mul_1Mul-dense_40/ActivityRegularizer/mul_1/x:output:0&dense_40/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2$
"dense_40/ActivityRegularizer/mul_1À
 dense_40/ActivityRegularizer/addAddV2$dense_40/ActivityRegularizer/mul:z:0&dense_40/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/add
"dense_40/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_40/ActivityRegularizer/Const¿
 dense_40/ActivityRegularizer/SumSum$dense_40/ActivityRegularizer/add:z:0+dense_40/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_40/ActivityRegularizer/Sum
$dense_40/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$dense_40/ActivityRegularizer/mul_2/xÊ
"dense_40/ActivityRegularizer/mul_2Mul-dense_40/ActivityRegularizer/mul_2/x:output:0)dense_40/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_40/ActivityRegularizer/mul_2
"dense_40/ActivityRegularizer/ShapeShapedense_40/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_40/ActivityRegularizer/Shape®
0dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_40/ActivityRegularizer/strided_slice/stack²
2dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_1²
2dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_2
*dense_40/ActivityRegularizer/strided_sliceStridedSlice+dense_40/ActivityRegularizer/Shape:output:09dense_40/ActivityRegularizer/strided_slice/stack:output:0;dense_40/ActivityRegularizer/strided_slice/stack_1:output:0;dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_40/ActivityRegularizer/strided_slice³
!dense_40/ActivityRegularizer/CastCast3dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_40/ActivityRegularizer/CastË
&dense_40/ActivityRegularizer/truediv_2RealDiv&dense_40/ActivityRegularizer/mul_2:z:0%dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_40/ActivityRegularizer/truediv_2Î
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulß
IdentityIdentitydense_40/Sigmoid:y:0 ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityè

Identity_1Identity*dense_40/ActivityRegularizer/truediv_2:z:0 ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
¬

0__inference_sequential_40_layer_call_fn_16601409

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
K__inference_sequential_40_layer_call_and_return_conditional_losses_166008482
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


K__inference_sequential_41_layer_call_and_return_conditional_losses_16600994

inputs#
dense_41_16600982: ^
dense_41_16600984:^
identity¢ dense_41/StatefulPartitionedCall¢1dense_41/kernel/Regularizer/Square/ReadVariableOp
 dense_41/StatefulPartitionedCallStatefulPartitionedCallinputsdense_41_16600982dense_41_16600984*
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
F__inference_dense_41_layer_call_and_return_conditional_losses_166009382"
 dense_41/StatefulPartitionedCall¸
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_41_16600982*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulÔ
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_41/StatefulPartitionedCall2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ä
³
__inference_loss_fn_0_16601648L
:dense_40_kernel_regularizer_square_readvariableop_resource:^ 
identity¢1dense_40/kernel/Regularizer/Square/ReadVariableOpá
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_40_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul
IdentityIdentity#dense_40/kernel/Regularizer/mul:z:02^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp


+__inference_dense_41_layer_call_fn_16601663

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
F__inference_dense_41_layer_call_and_return_conditional_losses_166009382
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_16600848

inputs#
dense_40_16600827:^ 
dense_40_16600829: 
identity

identity_1¢ dense_40/StatefulPartitionedCall¢1dense_40/kernel/Regularizer/Square/ReadVariableOp
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_16600827dense_40_16600829*
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
F__inference_dense_40_layer_call_and_return_conditional_losses_166007602"
 dense_40/StatefulPartitionedCallü
,dense_40/ActivityRegularizer/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
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
2__inference_dense_40_activity_regularizer_166007362.
,dense_40/ActivityRegularizer/PartitionedCall¡
"dense_40/ActivityRegularizer/ShapeShape)dense_40/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_40/ActivityRegularizer/Shape®
0dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_40/ActivityRegularizer/strided_slice/stack²
2dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_1²
2dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_2
*dense_40/ActivityRegularizer/strided_sliceStridedSlice+dense_40/ActivityRegularizer/Shape:output:09dense_40/ActivityRegularizer/strided_slice/stack:output:0;dense_40/ActivityRegularizer/strided_slice/stack_1:output:0;dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_40/ActivityRegularizer/strided_slice³
!dense_40/ActivityRegularizer/CastCast3dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_40/ActivityRegularizer/CastÖ
$dense_40/ActivityRegularizer/truedivRealDiv5dense_40/ActivityRegularizer/PartitionedCall:output:0%dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_40/ActivityRegularizer/truediv¸
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_16600827*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulÔ
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_40/ActivityRegularizer/truediv:z:0!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
²

0__inference_sequential_40_layer_call_fn_16600866
input_21
unknown:^ 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0*
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_166008482
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
input_21
Êe
º
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601324
xG
5sequential_40_dense_40_matmul_readvariableop_resource:^ D
6sequential_40_dense_40_biasadd_readvariableop_resource: G
5sequential_41_dense_41_matmul_readvariableop_resource: ^D
6sequential_41_dense_41_biasadd_readvariableop_resource:^
identity

identity_1¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢-sequential_40/dense_40/BiasAdd/ReadVariableOp¢,sequential_40/dense_40/MatMul/ReadVariableOp¢-sequential_41/dense_41/BiasAdd/ReadVariableOp¢,sequential_41/dense_41/MatMul/ReadVariableOpÒ
,sequential_40/dense_40/MatMul/ReadVariableOpReadVariableOp5sequential_40_dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02.
,sequential_40/dense_40/MatMul/ReadVariableOp³
sequential_40/dense_40/MatMulMatMulx4sequential_40/dense_40/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_40/dense_40/MatMulÑ
-sequential_40/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_40_dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_40/dense_40/BiasAdd/ReadVariableOpÝ
sequential_40/dense_40/BiasAddBiasAdd'sequential_40/dense_40/MatMul:product:05sequential_40/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_40/dense_40/BiasAdd¦
sequential_40/dense_40/SigmoidSigmoid'sequential_40/dense_40/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_40/dense_40/SigmoidÈ
Asequential_40/dense_40/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_40/dense_40/ActivityRegularizer/Mean/reduction_indicesÿ
/sequential_40/dense_40/ActivityRegularizer/MeanMean"sequential_40/dense_40/Sigmoid:y:0Jsequential_40/dense_40/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 21
/sequential_40/dense_40/ActivityRegularizer/Mean±
4sequential_40/dense_40/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÿæÛ.26
4sequential_40/dense_40/ActivityRegularizer/Maximum/y
2sequential_40/dense_40/ActivityRegularizer/MaximumMaximum8sequential_40/dense_40/ActivityRegularizer/Mean:output:0=sequential_40/dense_40/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 24
2sequential_40/dense_40/ActivityRegularizer/Maximum±
4sequential_40/dense_40/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<26
4sequential_40/dense_40/ActivityRegularizer/truediv/x
2sequential_40/dense_40/ActivityRegularizer/truedivRealDiv=sequential_40/dense_40/ActivityRegularizer/truediv/x:output:06sequential_40/dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 24
2sequential_40/dense_40/ActivityRegularizer/truedivÄ
.sequential_40/dense_40/ActivityRegularizer/LogLog6sequential_40/dense_40/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/Log©
0sequential_40/dense_40/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<22
0sequential_40/dense_40/ActivityRegularizer/mul/xû
.sequential_40/dense_40/ActivityRegularizer/mulMul9sequential_40/dense_40/ActivityRegularizer/mul/x:output:02sequential_40/dense_40/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/mul©
0sequential_40/dense_40/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_40/dense_40/ActivityRegularizer/sub/xÿ
.sequential_40/dense_40/ActivityRegularizer/subSub9sequential_40/dense_40/ActivityRegularizer/sub/x:output:06sequential_40/dense_40/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/subµ
6sequential_40/dense_40/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?28
6sequential_40/dense_40/ActivityRegularizer/truediv_1/x
4sequential_40/dense_40/ActivityRegularizer/truediv_1RealDiv?sequential_40/dense_40/ActivityRegularizer/truediv_1/x:output:02sequential_40/dense_40/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 26
4sequential_40/dense_40/ActivityRegularizer/truediv_1Ê
0sequential_40/dense_40/ActivityRegularizer/Log_1Log8sequential_40/dense_40/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 22
0sequential_40/dense_40/ActivityRegularizer/Log_1­
2sequential_40/dense_40/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *¤p}?24
2sequential_40/dense_40/ActivityRegularizer/mul_1/x
0sequential_40/dense_40/ActivityRegularizer/mul_1Mul;sequential_40/dense_40/ActivityRegularizer/mul_1/x:output:04sequential_40/dense_40/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 22
0sequential_40/dense_40/ActivityRegularizer/mul_1ø
.sequential_40/dense_40/ActivityRegularizer/addAddV22sequential_40/dense_40/ActivityRegularizer/mul:z:04sequential_40/dense_40/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/add®
0sequential_40/dense_40/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_40/dense_40/ActivityRegularizer/Const÷
.sequential_40/dense_40/ActivityRegularizer/SumSum2sequential_40/dense_40/ActivityRegularizer/add:z:09sequential_40/dense_40/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_40/dense_40/ActivityRegularizer/Sum­
2sequential_40/dense_40/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_40/dense_40/ActivityRegularizer/mul_2/x
0sequential_40/dense_40/ActivityRegularizer/mul_2Mul;sequential_40/dense_40/ActivityRegularizer/mul_2/x:output:07sequential_40/dense_40/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_40/dense_40/ActivityRegularizer/mul_2¶
0sequential_40/dense_40/ActivityRegularizer/ShapeShape"sequential_40/dense_40/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_40/dense_40/ActivityRegularizer/ShapeÊ
>sequential_40/dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_40/dense_40/ActivityRegularizer/strided_slice/stackÎ
@sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1Î
@sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2ä
8sequential_40/dense_40/ActivityRegularizer/strided_sliceStridedSlice9sequential_40/dense_40/ActivityRegularizer/Shape:output:0Gsequential_40/dense_40/ActivityRegularizer/strided_slice/stack:output:0Isequential_40/dense_40/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_40/dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_40/dense_40/ActivityRegularizer/strided_sliceÝ
/sequential_40/dense_40/ActivityRegularizer/CastCastAsequential_40/dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_40/dense_40/ActivityRegularizer/Cast
4sequential_40/dense_40/ActivityRegularizer/truediv_2RealDiv4sequential_40/dense_40/ActivityRegularizer/mul_2:z:03sequential_40/dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_40/dense_40/ActivityRegularizer/truediv_2Ò
,sequential_41/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_41_dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02.
,sequential_41/dense_41/MatMul/ReadVariableOpÔ
sequential_41/dense_41/MatMulMatMul"sequential_40/dense_40/Sigmoid:y:04sequential_41/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
sequential_41/dense_41/MatMulÑ
-sequential_41/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_41_dense_41_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02/
-sequential_41/dense_41/BiasAdd/ReadVariableOpÝ
sequential_41/dense_41/BiasAddBiasAdd'sequential_41/dense_41/MatMul:product:05sequential_41/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_41/dense_41/BiasAdd¦
sequential_41/dense_41/SigmoidSigmoid'sequential_41/dense_41/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2 
sequential_41/dense_41/SigmoidÜ
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_40_dense_40_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulÜ
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_41_dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mul
IdentityIdentity"sequential_41/dense_41/Sigmoid:y:02^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp.^sequential_40/dense_40/BiasAdd/ReadVariableOp-^sequential_40/dense_40/MatMul/ReadVariableOp.^sequential_41/dense_41/BiasAdd/ReadVariableOp-^sequential_41/dense_41/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity¥

Identity_1Identity8sequential_40/dense_40/ActivityRegularizer/truediv_2:z:02^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp.^sequential_40/dense_40/BiasAdd/ReadVariableOp-^sequential_40/dense_40/MatMul/ReadVariableOp.^sequential_41/dense_41/BiasAdd/ReadVariableOp-^sequential_41/dense_41/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_40/dense_40/BiasAdd/ReadVariableOp-sequential_40/dense_40/BiasAdd/ReadVariableOp2\
,sequential_40/dense_40/MatMul/ReadVariableOp,sequential_40/dense_40/MatMul/ReadVariableOp2^
-sequential_41/dense_41/BiasAdd/ReadVariableOp-sequential_41/dense_41/BiasAdd/ReadVariableOp2\
,sequential_41/dense_41/MatMul/ReadVariableOp,sequential_41/dense_41/MatMul/ReadVariableOp:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
ó
Ï
1__inference_autoencoder_20_layer_call_fn_16601265
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_166011282
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

Õ
1__inference_autoencoder_20_layer_call_fn_16601084
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_166010722
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
ô"

K__inference_sequential_40_layer_call_and_return_conditional_losses_16600914
input_21#
dense_40_16600893:^ 
dense_40_16600895: 
identity

identity_1¢ dense_40/StatefulPartitionedCall¢1dense_40/kernel/Regularizer/Square/ReadVariableOp
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinput_21dense_40_16600893dense_40_16600895*
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
F__inference_dense_40_layer_call_and_return_conditional_losses_166007602"
 dense_40/StatefulPartitionedCallü
,dense_40/ActivityRegularizer/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
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
2__inference_dense_40_activity_regularizer_166007362.
,dense_40/ActivityRegularizer/PartitionedCall¡
"dense_40/ActivityRegularizer/ShapeShape)dense_40/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_40/ActivityRegularizer/Shape®
0dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_40/ActivityRegularizer/strided_slice/stack²
2dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_1²
2dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_2
*dense_40/ActivityRegularizer/strided_sliceStridedSlice+dense_40/ActivityRegularizer/Shape:output:09dense_40/ActivityRegularizer/strided_slice/stack:output:0;dense_40/ActivityRegularizer/strided_slice/stack_1:output:0;dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_40/ActivityRegularizer/strided_slice³
!dense_40/ActivityRegularizer/CastCast3dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_40/ActivityRegularizer/CastÖ
$dense_40/ActivityRegularizer/truedivRealDiv5dense_40/ActivityRegularizer/PartitionedCall:output:0%dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_40/ActivityRegularizer/truediv¸
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_16600893*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulÔ
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_40/ActivityRegularizer/truediv:z:0!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
"
_user_specified_name
input_21
ò$
Î
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601128
x(
sequential_40_16601103:^ $
sequential_40_16601105: (
sequential_41_16601109: ^$
sequential_41_16601111:^
identity

identity_1¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢%sequential_40/StatefulPartitionedCall¢%sequential_41/StatefulPartitionedCall±
%sequential_40/StatefulPartitionedCallStatefulPartitionedCallxsequential_40_16601103sequential_40_16601105*
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_166008482'
%sequential_40/StatefulPartitionedCallÛ
%sequential_41/StatefulPartitionedCallStatefulPartitionedCall.sequential_40/StatefulPartitionedCall:output:0sequential_41_16601109sequential_41_16601111*
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_166009942'
%sequential_41/StatefulPartitionedCall½
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_40_16601103*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul½
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_41_16601109*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulº
IdentityIdentity.sequential_41/StatefulPartitionedCall:output:02^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp&^sequential_40/StatefulPartitionedCall&^sequential_41/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_40/StatefulPartitionedCall:output:12^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp&^sequential_40/StatefulPartitionedCall&^sequential_41/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_40/StatefulPartitionedCall%sequential_40/StatefulPartitionedCall2N
%sequential_41/StatefulPartitionedCall%sequential_41/StatefulPartitionedCall:J F
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^

_user_specified_nameX
©

0__inference_sequential_41_layer_call_fn_16601525

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
K__inference_sequential_41_layer_call_and_return_conditional_losses_166009512
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
Ç
Ü
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601611
dense_41_input9
'dense_41_matmul_readvariableop_resource: ^6
(dense_41_biasadd_readvariableop_resource:^
identity¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¨
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_41/MatMul/ReadVariableOp
dense_41/MatMulMatMuldense_41_input&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/MatMul§
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_41/BiasAdd/ReadVariableOp¥
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/BiasAdd|
dense_41/SigmoidSigmoiddense_41/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/SigmoidÎ
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulß
IdentityIdentitydense_41/Sigmoid:y:0 ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:W S
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_namedense_41_input
î"

K__inference_sequential_40_layer_call_and_return_conditional_losses_16600782

inputs#
dense_40_16600761:^ 
dense_40_16600763: 
identity

identity_1¢ dense_40/StatefulPartitionedCall¢1dense_40/kernel/Regularizer/Square/ReadVariableOp
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_16600761dense_40_16600763*
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
F__inference_dense_40_layer_call_and_return_conditional_losses_166007602"
 dense_40/StatefulPartitionedCallü
,dense_40/ActivityRegularizer/PartitionedCallPartitionedCall)dense_40/StatefulPartitionedCall:output:0*
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
2__inference_dense_40_activity_regularizer_166007362.
,dense_40/ActivityRegularizer/PartitionedCall¡
"dense_40/ActivityRegularizer/ShapeShape)dense_40/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_40/ActivityRegularizer/Shape®
0dense_40/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_40/ActivityRegularizer/strided_slice/stack²
2dense_40/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_1²
2dense_40/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_40/ActivityRegularizer/strided_slice/stack_2
*dense_40/ActivityRegularizer/strided_sliceStridedSlice+dense_40/ActivityRegularizer/Shape:output:09dense_40/ActivityRegularizer/strided_slice/stack:output:0;dense_40/ActivityRegularizer/strided_slice/stack_1:output:0;dense_40/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_40/ActivityRegularizer/strided_slice³
!dense_40/ActivityRegularizer/CastCast3dense_40/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_40/ActivityRegularizer/CastÖ
$dense_40/ActivityRegularizer/truedivRealDiv5dense_40/ActivityRegularizer/PartitionedCall:output:0%dense_40/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_40/ActivityRegularizer/truediv¸
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_40_16600761*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mulÔ
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

IdentityÆ

Identity_1Identity(dense_40/ActivityRegularizer/truediv:z:0!^dense_40/StatefulPartitionedCall2^dense_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : 2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
 
_user_specified_nameinputs
Ç
ª
!__inference__traced_save_16601743
file_prefix.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601182
input_1(
sequential_40_16601157:^ $
sequential_40_16601159: (
sequential_41_16601163: ^$
sequential_41_16601165:^
identity

identity_1¢1dense_40/kernel/Regularizer/Square/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¢%sequential_40/StatefulPartitionedCall¢%sequential_41/StatefulPartitionedCall·
%sequential_40/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_40_16601157sequential_40_16601159*
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_166007822'
%sequential_40/StatefulPartitionedCallÛ
%sequential_41/StatefulPartitionedCallStatefulPartitionedCall.sequential_40/StatefulPartitionedCall:output:0sequential_41_16601163sequential_41_16601165*
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_166009512'
%sequential_41/StatefulPartitionedCall½
1dense_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_40_16601157*
_output_shapes

:^ *
dtype023
1dense_40/kernel/Regularizer/Square/ReadVariableOp¶
"dense_40/kernel/Regularizer/SquareSquare9dense_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2$
"dense_40/kernel/Regularizer/Square
!dense_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_40/kernel/Regularizer/Const¾
dense_40/kernel/Regularizer/SumSum&dense_40/kernel/Regularizer/Square:y:0*dense_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/Sum
!dense_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_40/kernel/Regularizer/mul/xÀ
dense_40/kernel/Regularizer/mulMul*dense_40/kernel/Regularizer/mul/x:output:0(dense_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_40/kernel/Regularizer/mul½
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_41_16601163*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulº
IdentityIdentity.sequential_41/StatefulPartitionedCall:output:02^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp&^sequential_40/StatefulPartitionedCall&^sequential_41/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity­

Identity_1Identity.sequential_40/StatefulPartitionedCall:output:12^dense_40/kernel/Regularizer/Square/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp&^sequential_40/StatefulPartitionedCall&^sequential_41/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ^: : : : 2f
1dense_40/kernel/Regularizer/Square/ReadVariableOp1dense_40/kernel/Regularizer/Square/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_40/StatefulPartitionedCall%sequential_40/StatefulPartitionedCall2N
%sequential_41/StatefulPartitionedCall%sequential_41/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
!
_user_specified_name	input_1
¯
Ô
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601560

inputs9
'dense_41_matmul_readvariableop_resource: ^6
(dense_41_biasadd_readvariableop_resource:^
identity¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp¨
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02 
dense_41/MatMul/ReadVariableOp
dense_41/MatMulMatMulinputs&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/MatMul§
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02!
dense_41/BiasAdd/ReadVariableOp¥
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/BiasAdd|
dense_41/SigmoidSigmoiddense_41/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2
dense_41/SigmoidÎ
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulß
IdentityIdentitydense_41/Sigmoid:y:0 ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2f
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
­
«
F__inference_dense_41_layer_call_and_return_conditional_losses_16601680

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_41/kernel/Regularizer/Square/ReadVariableOp
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
1dense_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype023
1dense_41/kernel/Regularizer/Square/ReadVariableOp¶
"dense_41/kernel/Regularizer/SquareSquare9dense_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2$
"dense_41/kernel/Regularizer/Square
!dense_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_41/kernel/Regularizer/Const¾
dense_41/kernel/Regularizer/SumSum&dense_41/kernel/Regularizer/Square:y:0*dense_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/Sum
!dense_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!dense_41/kernel/Regularizer/mul/xÀ
dense_41/kernel/Regularizer/mulMul*dense_41/kernel/Regularizer/mul/x:output:0(dense_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_41/kernel/Regularizer/mulÄ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_41/kernel/Regularizer/Square/ReadVariableOp*
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
1dense_41/kernel/Regularizer/Square/ReadVariableOp1dense_41/kernel/Regularizer/Square/ReadVariableOp:O K
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
_tf_keras_model{"name": "autoencoder_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
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
_tf_keras_sequentialÞ{"name": "sequential_40", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_40", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_21"}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_21"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_40", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_21"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
¾
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_sequentialé{"name": "sequential_41", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_41", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_41_input"}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_41_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_41", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_41_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
{"name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
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
_tf_keras_layer®{"name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
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
!:^ 2dense_40/kernel
: 2dense_40/bias
!: ^2dense_41/kernel
:^2dense_41/bias
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
1__inference_autoencoder_20_layer_call_fn_16601084
1__inference_autoencoder_20_layer_call_fn_16601251
1__inference_autoencoder_20_layer_call_fn_16601265
1__inference_autoencoder_20_layer_call_fn_16601154®
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
#__inference__wrapped_model_16600707¶
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601324
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601383
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601182
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601210®
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
0__inference_sequential_40_layer_call_fn_16600790
0__inference_sequential_40_layer_call_fn_16601399
0__inference_sequential_40_layer_call_fn_16601409
0__inference_sequential_40_layer_call_fn_16600866À
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_16601455
K__inference_sequential_40_layer_call_and_return_conditional_losses_16601501
K__inference_sequential_40_layer_call_and_return_conditional_losses_16600890
K__inference_sequential_40_layer_call_and_return_conditional_losses_16600914À
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
0__inference_sequential_41_layer_call_fn_16601516
0__inference_sequential_41_layer_call_fn_16601525
0__inference_sequential_41_layer_call_fn_16601534
0__inference_sequential_41_layer_call_fn_16601543À
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601560
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601577
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601594
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601611À
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
&__inference_signature_wrapper_16601237input_1"
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
+__inference_dense_40_layer_call_fn_16601626¢
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
J__inference_dense_40_layer_call_and_return_all_conditional_losses_16601637¢
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
__inference_loss_fn_0_16601648
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
+__inference_dense_41_layer_call_fn_16601663¢
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
F__inference_dense_41_layer_call_and_return_conditional_losses_16601680¢
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
__inference_loss_fn_1_16601691
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
2__inference_dense_40_activity_regularizer_16600736²
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
F__inference_dense_40_layer_call_and_return_conditional_losses_16601708¢
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
#__inference__wrapped_model_16600707m0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ^
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ^Á
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601182q4¢1
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601210q4¢1
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601324k.¢+
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
L__inference_autoencoder_20_layer_call_and_return_conditional_losses_16601383k.¢+
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
1__inference_autoencoder_20_layer_call_fn_16601084V4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p 
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_20_layer_call_fn_16601154V4¢1
*¢'
!
input_1ÿÿÿÿÿÿÿÿÿ^
p
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_20_layer_call_fn_16601251P.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p 
ª "ÿÿÿÿÿÿÿÿÿ^
1__inference_autoencoder_20_layer_call_fn_16601265P.¢+
$¢!

Xÿÿÿÿÿÿÿÿÿ^
p
ª "ÿÿÿÿÿÿÿÿÿ^e
2__inference_dense_40_activity_regularizer_16600736/$¢!
¢


activation
ª " ¸
J__inference_dense_40_layer_call_and_return_all_conditional_losses_16601637j/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 ¦
F__inference_dense_40_layer_call_and_return_conditional_losses_16601708\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 ~
+__inference_dense_40_layer_call_fn_16601626O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ^
ª "ÿÿÿÿÿÿÿÿÿ ¦
F__inference_dense_41_layer_call_and_return_conditional_losses_16601680\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 ~
+__inference_dense_41_layer_call_fn_16601663O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ^=
__inference_loss_fn_0_16601648¢

¢ 
ª " =
__inference_loss_fn_1_16601691¢

¢ 
ª " Ã
K__inference_sequential_40_layer_call_and_return_conditional_losses_16600890t9¢6
/¢,
"
input_21ÿÿÿÿÿÿÿÿÿ^
p 

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Ã
K__inference_sequential_40_layer_call_and_return_conditional_losses_16600914t9¢6
/¢,
"
input_21ÿÿÿÿÿÿÿÿÿ^
p

 
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ 

	
1/0 Á
K__inference_sequential_40_layer_call_and_return_conditional_losses_16601455r7¢4
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
K__inference_sequential_40_layer_call_and_return_conditional_losses_16601501r7¢4
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
0__inference_sequential_40_layer_call_fn_16600790Y9¢6
/¢,
"
input_21ÿÿÿÿÿÿÿÿÿ^
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_40_layer_call_fn_16600866Y9¢6
/¢,
"
input_21ÿÿÿÿÿÿÿÿÿ^
p

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_40_layer_call_fn_16601399W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_40_layer_call_fn_16601409W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ^
p

 
ª "ÿÿÿÿÿÿÿÿÿ ³
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601560d7¢4
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601577d7¢4
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
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601594l?¢<
5¢2
(%
dense_41_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 »
K__inference_sequential_41_layer_call_and_return_conditional_losses_16601611l?¢<
5¢2
(%
dense_41_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ^
 
0__inference_sequential_41_layer_call_fn_16601516_?¢<
5¢2
(%
dense_41_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_41_layer_call_fn_16601525W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_41_layer_call_fn_16601534W7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ^
0__inference_sequential_41_layer_call_fn_16601543_?¢<
5¢2
(%
dense_41_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "ÿÿÿÿÿÿÿÿÿ^¢
&__inference_signature_wrapper_16601237x;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ^"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ^