??

??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??	
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
??*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:?*
dtype0
|
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_17/kernel
u
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel* 
_output_shapes
:
??*
dtype0
s
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_17/bias
l
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
history
encoder
decoder
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
y
	layer_with_weights-0
	layer-0

	variables
trainable_variables
regularization_losses
	keras_api
y
layer_with_weights-0
layer-0
	variables
trainable_variables
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
?
non_trainable_variables
layer_regularization_losses
	variables
metrics
trainable_variables

layers
layer_metrics
regularization_losses
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api

0
1

0
1
 
?
 non_trainable_variables
!layer_regularization_losses

	variables
"metrics
trainable_variables

#layers
$layer_metrics
regularization_losses
h

kernel
bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api

0
1

0
1
 
?
)non_trainable_variables
*layer_regularization_losses
	variables
+metrics
trainable_variables

,layers
-layer_metrics
regularization_losses
KI
VARIABLE_VALUEdense_16/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_16/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_17/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_17/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
?
.layer_regularization_losses
	variables
/metrics
trainable_variables
0layer_metrics

1layers
2non_trainable_variables
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
?
3layer_regularization_losses
%	variables
4metrics
&trainable_variables
5layer_metrics

6layers
7non_trainable_variables
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
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_4580376
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpConst*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_4580882
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_4580904??
?A
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580594

inputs;
'dense_16_matmul_readvariableop_resource:
??7
(dense_16_biasadd_readvariableop_resource:	?
identity

identity_1??dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?1dense_16/kernel/Regularizer/Square/ReadVariableOp?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulinputs&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/BiasAdd}
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_16/Sigmoid?
3dense_16/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_16/ActivityRegularizer/Mean/reduction_indices?
!dense_16/ActivityRegularizer/MeanMeandense_16/Sigmoid:y:0<dense_16/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2#
!dense_16/ActivityRegularizer/Mean?
&dense_16/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2(
&dense_16/ActivityRegularizer/Maximum/y?
$dense_16/ActivityRegularizer/MaximumMaximum*dense_16/ActivityRegularizer/Mean:output:0/dense_16/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2&
$dense_16/ActivityRegularizer/Maximum?
&dense_16/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2(
&dense_16/ActivityRegularizer/truediv/x?
$dense_16/ActivityRegularizer/truedivRealDiv/dense_16/ActivityRegularizer/truediv/x:output:0(dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2&
$dense_16/ActivityRegularizer/truediv?
 dense_16/ActivityRegularizer/LogLog(dense_16/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2"
 dense_16/ActivityRegularizer/Log?
"dense_16/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_16/ActivityRegularizer/mul/x?
 dense_16/ActivityRegularizer/mulMul+dense_16/ActivityRegularizer/mul/x:output:0$dense_16/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2"
 dense_16/ActivityRegularizer/mul?
"dense_16/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"dense_16/ActivityRegularizer/sub/x?
 dense_16/ActivityRegularizer/subSub+dense_16/ActivityRegularizer/sub/x:output:0(dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2"
 dense_16/ActivityRegularizer/sub?
(dense_16/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2*
(dense_16/ActivityRegularizer/truediv_1/x?
&dense_16/ActivityRegularizer/truediv_1RealDiv1dense_16/ActivityRegularizer/truediv_1/x:output:0$dense_16/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2(
&dense_16/ActivityRegularizer/truediv_1?
"dense_16/ActivityRegularizer/Log_1Log*dense_16/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2$
"dense_16/ActivityRegularizer/Log_1?
$dense_16/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2&
$dense_16/ActivityRegularizer/mul_1/x?
"dense_16/ActivityRegularizer/mul_1Mul-dense_16/ActivityRegularizer/mul_1/x:output:0&dense_16/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2$
"dense_16/ActivityRegularizer/mul_1?
 dense_16/ActivityRegularizer/addAddV2$dense_16/ActivityRegularizer/mul:z:0&dense_16/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2"
 dense_16/ActivityRegularizer/add?
"dense_16/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_16/ActivityRegularizer/Const?
 dense_16/ActivityRegularizer/SumSum$dense_16/ActivityRegularizer/add:z:0+dense_16/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_16/ActivityRegularizer/Sum?
$dense_16/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$dense_16/ActivityRegularizer/mul_2/x?
"dense_16/ActivityRegularizer/mul_2Mul-dense_16/ActivityRegularizer/mul_2/x:output:0)dense_16/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_16/ActivityRegularizer/mul_2?
"dense_16/ActivityRegularizer/ShapeShapedense_16/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape?
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack?
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1?
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2?
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice?
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Cast?
&dense_16/ActivityRegularizer/truediv_2RealDiv&dense_16/ActivityRegularizer/mul_2:z:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_16/ActivityRegularizer/truediv_2?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentitydense_16/Sigmoid:y:0 ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity*dense_16/ActivityRegularizer/truediv_2:z:0 ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_16_layer_call_fn_4579929
input_9
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_45799212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_9
?
?
*__inference_dense_17_layer_call_fn_4580819

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_45800772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_dense_16_layer_call_and_return_all_conditional_losses_4580767

inputs
unknown:
??
	unknown_0:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_45798992
StatefulPartitionedCall?
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
GPU 2J 8? *:
f5R3
1__inference_dense_16_activity_regularizer_45798752
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580053
input_9$
dense_16_4580032:
??
dense_16_4580034:	?
identity

identity_1?? dense_16/StatefulPartitionedCall?1dense_16/kernel/Regularizer/Square/ReadVariableOp?
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinput_9dense_16_4580032dense_16_4580034*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_45798992"
 dense_16/StatefulPartitionedCall?
,dense_16/ActivityRegularizer/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *:
f5R3
1__inference_dense_16_activity_regularizer_45798752.
,dense_16/ActivityRegularizer/PartitionedCall?
"dense_16/ActivityRegularizer/ShapeShape)dense_16/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape?
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack?
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1?
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2?
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice?
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Cast?
$dense_16/ActivityRegularizer/truedivRealDiv5dense_16/ActivityRegularizer/PartitionedCall:output:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truediv?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_4580032* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity(dense_16/ActivityRegularizer/truediv:z:0!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_9
?%
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580321
input_1)
sequential_16_4580296:
??$
sequential_16_4580298:	?)
sequential_17_4580302:
??$
sequential_17_4580304:	?
identity

identity_1??1dense_16/kernel/Regularizer/Square/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_16_4580296sequential_16_4580298*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_45799212'
%sequential_16/StatefulPartitionedCall?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_4580302sequential_17_4580304*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_45800902'
%sequential_17/StatefulPartitionedCall?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_4580296* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_4580302* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:02^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity.sequential_16/StatefulPartitionedCall:output:12^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
__inference_loss_fn_1_4580830N
:dense_17_kernel_regularizer_square_readvariableop_resource:
??
identity??1dense_17/kernel/Regularizer/Square/ReadVariableOp?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_17_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity#dense_17/kernel/Regularizer/mul:z:02^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp
?
?
 __inference__traced_save_4580882
file_prefix.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*=
_input_shapes,
*: :
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?
?
/__inference_autoencoder_8_layer_call_fn_4580293
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_45802672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
E__inference_dense_16_layer_call_and_return_conditional_losses_4579899

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_16/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_16_layer_call_fn_4580548

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_45799872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_16_layer_call_fn_4580005
input_9
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_45799872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_9
?
?
/__inference_autoencoder_8_layer_call_fn_4580390
x
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_45802112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?"
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580029
input_9$
dense_16_4580008:
??
dense_16_4580010:	?
identity

identity_1?? dense_16/StatefulPartitionedCall?1dense_16/kernel/Regularizer/Square/ReadVariableOp?
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinput_9dense_16_4580008dense_16_4580010*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_45798992"
 dense_16/StatefulPartitionedCall?
,dense_16/ActivityRegularizer/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *:
f5R3
1__inference_dense_16_activity_regularizer_45798752.
,dense_16/ActivityRegularizer/PartitionedCall?
"dense_16/ActivityRegularizer/ShapeShape)dense_16/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape?
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack?
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1?
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2?
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice?
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Cast?
$dense_16/ActivityRegularizer/truedivRealDiv5dense_16/ActivityRegularizer/PartitionedCall:output:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truediv?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_4580008* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity(dense_16/ActivityRegularizer/truediv:z:0!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_9
?
?
/__inference_autoencoder_8_layer_call_fn_4580404
x
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_45802672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
/__inference_sequential_17_layer_call_fn_4580664

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_45800902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_17_layer_call_and_return_conditional_losses_4580077

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_17_layer_call_fn_4580673

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_45801332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_16_layer_call_fn_4580538

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_45799212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_17_layer_call_fn_4580655
dense_17_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_17_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_45800902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_17_input
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580699

inputs;
'dense_17_matmul_readvariableop_resource:
??7
(dense_17_biasadd_readvariableop_resource:	?
identity??dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMulinputs&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/BiasAdd}
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_17/Sigmoid?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentitydense_17/Sigmoid:y:0 ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
"__inference__wrapped_model_4579846
input_1W
Cautoencoder_8_sequential_16_dense_16_matmul_readvariableop_resource:
??S
Dautoencoder_8_sequential_16_dense_16_biasadd_readvariableop_resource:	?W
Cautoencoder_8_sequential_17_dense_17_matmul_readvariableop_resource:
??S
Dautoencoder_8_sequential_17_dense_17_biasadd_readvariableop_resource:	?
identity??;autoencoder_8/sequential_16/dense_16/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_16/dense_16/MatMul/ReadVariableOp?;autoencoder_8/sequential_17/dense_17/BiasAdd/ReadVariableOp?:autoencoder_8/sequential_17/dense_17/MatMul/ReadVariableOp?
:autoencoder_8/sequential_16/dense_16/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_16_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:autoencoder_8/sequential_16/dense_16/MatMul/ReadVariableOp?
+autoencoder_8/sequential_16/dense_16/MatMulMatMulinput_1Bautoencoder_8/sequential_16/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+autoencoder_8/sequential_16/dense_16/MatMul?
;autoencoder_8/sequential_16/dense_16/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_16_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;autoencoder_8/sequential_16/dense_16/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_16/dense_16/BiasAddBiasAdd5autoencoder_8/sequential_16/dense_16/MatMul:product:0Cautoencoder_8/sequential_16/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_8/sequential_16/dense_16/BiasAdd?
,autoencoder_8/sequential_16/dense_16/SigmoidSigmoid5autoencoder_8/sequential_16/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_8/sequential_16/dense_16/Sigmoid?
Oautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2Q
Oautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Mean/reduction_indices?
=autoencoder_8/sequential_16/dense_16/ActivityRegularizer/MeanMean0autoencoder_8/sequential_16/dense_16/Sigmoid:y:0Xautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2?
=autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Mean?
Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2D
Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Maximum/y?
@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/MaximumMaximumFautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Mean:output:0Kautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2B
@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Maximum?
Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2D
Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv/x?
@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/truedivRealDivKautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv/x:output:0Dautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2B
@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv?
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/LogLogDautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2>
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Log?
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2@
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul/x?
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mulMulGautoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul/x:output:0@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2>
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul?
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2@
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/sub/x?
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/subSubGautoencoder_8/sequential_16/dense_16/ActivityRegularizer/sub/x:output:0Dautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2>
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/sub?
Dautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2F
Dautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv_1/x?
Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv_1RealDivMautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv_1/x:output:0@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2D
Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv_1?
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Log_1LogFautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2@
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Log_1?
@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2B
@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_1/x?
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_1MulIautoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_1/x:output:0Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2@
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_1?
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/addAddV2@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul:z:0Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2>
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/add?
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Const?
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/SumSum@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/add:z:0Gautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2>
<autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Sum?
@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_2/x?
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_2MulIautoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_2/x:output:0Eautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2@
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_2?
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/ShapeShape0autoencoder_8/sequential_16/dense_16/Sigmoid:y:0*
T0*
_output_shapes
:2@
>autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Shape?
Lautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2N
Lautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stack?
Nautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2P
Nautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1?
Nautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2P
Nautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2?
Fautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_sliceStridedSliceGautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Shape:output:0Uautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stack:output:0Wautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1:output:0Wautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2H
Fautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice?
=autoencoder_8/sequential_16/dense_16/ActivityRegularizer/CastCastOautoencoder_8/sequential_16/dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2?
=autoencoder_8/sequential_16/dense_16/ActivityRegularizer/Cast?
Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv_2RealDivBautoencoder_8/sequential_16/dense_16/ActivityRegularizer/mul_2:z:0Aautoencoder_8/sequential_16/dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2D
Bautoencoder_8/sequential_16/dense_16/ActivityRegularizer/truediv_2?
:autoencoder_8/sequential_17/dense_17/MatMul/ReadVariableOpReadVariableOpCautoencoder_8_sequential_17_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02<
:autoencoder_8/sequential_17/dense_17/MatMul/ReadVariableOp?
+autoencoder_8/sequential_17/dense_17/MatMulMatMul0autoencoder_8/sequential_16/dense_16/Sigmoid:y:0Bautoencoder_8/sequential_17/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2-
+autoencoder_8/sequential_17/dense_17/MatMul?
;autoencoder_8/sequential_17/dense_17/BiasAdd/ReadVariableOpReadVariableOpDautoencoder_8_sequential_17_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;autoencoder_8/sequential_17/dense_17/BiasAdd/ReadVariableOp?
,autoencoder_8/sequential_17/dense_17/BiasAddBiasAdd5autoencoder_8/sequential_17/dense_17/MatMul:product:0Cautoencoder_8/sequential_17/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_8/sequential_17/dense_17/BiasAdd?
,autoencoder_8/sequential_17/dense_17/SigmoidSigmoid5autoencoder_8/sequential_17/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2.
,autoencoder_8/sequential_17/dense_17/Sigmoid?
IdentityIdentity0autoencoder_8/sequential_17/dense_17/Sigmoid:y:0<^autoencoder_8/sequential_16/dense_16/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_16/dense_16/MatMul/ReadVariableOp<^autoencoder_8/sequential_17/dense_17/BiasAdd/ReadVariableOp;^autoencoder_8/sequential_17/dense_17/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2z
;autoencoder_8/sequential_16/dense_16/BiasAdd/ReadVariableOp;autoencoder_8/sequential_16/dense_16/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_16/dense_16/MatMul/ReadVariableOp:autoencoder_8/sequential_16/dense_16/MatMul/ReadVariableOp2z
;autoencoder_8/sequential_17/dense_17/BiasAdd/ReadVariableOp;autoencoder_8/sequential_17/dense_17/BiasAdd/ReadVariableOp2x
:autoencoder_8/sequential_17/dense_17/MatMul/ReadVariableOp:autoencoder_8/sequential_17/dense_17/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
*__inference_dense_16_layer_call_fn_4580776

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_45798992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580090

inputs$
dense_17_4580078:
??
dense_17_4580080:	?
identity?? dense_17/StatefulPartitionedCall?1dense_17/kernel/Regularizer/Square/ReadVariableOp?
 dense_17/StatefulPartitionedCallStatefulPartitionedCallinputsdense_17_4580078dense_17_4580080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_45800772"
 dense_17/StatefulPartitionedCall?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_17_4580078* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_17/StatefulPartitionedCall2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_autoencoder_8_layer_call_fn_4580223
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_45802112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?e
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580463
xI
5sequential_16_dense_16_matmul_readvariableop_resource:
??E
6sequential_16_dense_16_biasadd_readvariableop_resource:	?I
5sequential_17_dense_17_matmul_readvariableop_resource:
??E
6sequential_17_dense_17_biasadd_readvariableop_resource:	?
identity

identity_1??1dense_16/kernel/Regularizer/Square/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?-sequential_16/dense_16/BiasAdd/ReadVariableOp?,sequential_16/dense_16/MatMul/ReadVariableOp?-sequential_17/dense_17/BiasAdd/ReadVariableOp?,sequential_17/dense_17/MatMul/ReadVariableOp?
,sequential_16/dense_16/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_16/dense_16/MatMul/ReadVariableOp?
sequential_16/dense_16/MatMulMatMulx4sequential_16/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_16/dense_16/MatMul?
-sequential_16/dense_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_16/dense_16/BiasAdd/ReadVariableOp?
sequential_16/dense_16/BiasAddBiasAdd'sequential_16/dense_16/MatMul:product:05sequential_16/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_16/dense_16/BiasAdd?
sequential_16/dense_16/SigmoidSigmoid'sequential_16/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_16/dense_16/Sigmoid?
Asequential_16/dense_16/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_16/dense_16/ActivityRegularizer/Mean/reduction_indices?
/sequential_16/dense_16/ActivityRegularizer/MeanMean"sequential_16/dense_16/Sigmoid:y:0Jsequential_16/dense_16/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?21
/sequential_16/dense_16/ActivityRegularizer/Mean?
4sequential_16/dense_16/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.26
4sequential_16/dense_16/ActivityRegularizer/Maximum/y?
2sequential_16/dense_16/ActivityRegularizer/MaximumMaximum8sequential_16/dense_16/ActivityRegularizer/Mean:output:0=sequential_16/dense_16/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?24
2sequential_16/dense_16/ActivityRegularizer/Maximum?
4sequential_16/dense_16/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<26
4sequential_16/dense_16/ActivityRegularizer/truediv/x?
2sequential_16/dense_16/ActivityRegularizer/truedivRealDiv=sequential_16/dense_16/ActivityRegularizer/truediv/x:output:06sequential_16/dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?24
2sequential_16/dense_16/ActivityRegularizer/truediv?
.sequential_16/dense_16/ActivityRegularizer/LogLog6sequential_16/dense_16/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?20
.sequential_16/dense_16/ActivityRegularizer/Log?
0sequential_16/dense_16/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<22
0sequential_16/dense_16/ActivityRegularizer/mul/x?
.sequential_16/dense_16/ActivityRegularizer/mulMul9sequential_16/dense_16/ActivityRegularizer/mul/x:output:02sequential_16/dense_16/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?20
.sequential_16/dense_16/ActivityRegularizer/mul?
0sequential_16/dense_16/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_16/dense_16/ActivityRegularizer/sub/x?
.sequential_16/dense_16/ActivityRegularizer/subSub9sequential_16/dense_16/ActivityRegularizer/sub/x:output:06sequential_16/dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?20
.sequential_16/dense_16/ActivityRegularizer/sub?
6sequential_16/dense_16/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?28
6sequential_16/dense_16/ActivityRegularizer/truediv_1/x?
4sequential_16/dense_16/ActivityRegularizer/truediv_1RealDiv?sequential_16/dense_16/ActivityRegularizer/truediv_1/x:output:02sequential_16/dense_16/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?26
4sequential_16/dense_16/ActivityRegularizer/truediv_1?
0sequential_16/dense_16/ActivityRegularizer/Log_1Log8sequential_16/dense_16/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?22
0sequential_16/dense_16/ActivityRegularizer/Log_1?
2sequential_16/dense_16/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?24
2sequential_16/dense_16/ActivityRegularizer/mul_1/x?
0sequential_16/dense_16/ActivityRegularizer/mul_1Mul;sequential_16/dense_16/ActivityRegularizer/mul_1/x:output:04sequential_16/dense_16/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?22
0sequential_16/dense_16/ActivityRegularizer/mul_1?
.sequential_16/dense_16/ActivityRegularizer/addAddV22sequential_16/dense_16/ActivityRegularizer/mul:z:04sequential_16/dense_16/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?20
.sequential_16/dense_16/ActivityRegularizer/add?
0sequential_16/dense_16/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_16/dense_16/ActivityRegularizer/Const?
.sequential_16/dense_16/ActivityRegularizer/SumSum2sequential_16/dense_16/ActivityRegularizer/add:z:09sequential_16/dense_16/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_16/dense_16/ActivityRegularizer/Sum?
2sequential_16/dense_16/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_16/dense_16/ActivityRegularizer/mul_2/x?
0sequential_16/dense_16/ActivityRegularizer/mul_2Mul;sequential_16/dense_16/ActivityRegularizer/mul_2/x:output:07sequential_16/dense_16/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_16/dense_16/ActivityRegularizer/mul_2?
0sequential_16/dense_16/ActivityRegularizer/ShapeShape"sequential_16/dense_16/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_16/dense_16/ActivityRegularizer/Shape?
>sequential_16/dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_16/dense_16/ActivityRegularizer/strided_slice/stack?
@sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1?
@sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2?
8sequential_16/dense_16/ActivityRegularizer/strided_sliceStridedSlice9sequential_16/dense_16/ActivityRegularizer/Shape:output:0Gsequential_16/dense_16/ActivityRegularizer/strided_slice/stack:output:0Isequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_16/dense_16/ActivityRegularizer/strided_slice?
/sequential_16/dense_16/ActivityRegularizer/CastCastAsequential_16/dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_16/dense_16/ActivityRegularizer/Cast?
4sequential_16/dense_16/ActivityRegularizer/truediv_2RealDiv4sequential_16/dense_16/ActivityRegularizer/mul_2:z:03sequential_16/dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_16/dense_16/ActivityRegularizer/truediv_2?
,sequential_17/dense_17/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_17/dense_17/MatMul/ReadVariableOp?
sequential_17/dense_17/MatMulMatMul"sequential_16/dense_16/Sigmoid:y:04sequential_17/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_17/dense_17/MatMul?
-sequential_17/dense_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_17/dense_17/BiasAdd/ReadVariableOp?
sequential_17/dense_17/BiasAddBiasAdd'sequential_17/dense_17/MatMul:product:05sequential_17/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_17/BiasAdd?
sequential_17/dense_17/SigmoidSigmoid'sequential_17/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_17/Sigmoid?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity"sequential_17/dense_17/Sigmoid:y:02^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp.^sequential_16/dense_16/BiasAdd/ReadVariableOp-^sequential_16/dense_16/MatMul/ReadVariableOp.^sequential_17/dense_17/BiasAdd/ReadVariableOp-^sequential_17/dense_17/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity8sequential_16/dense_16/ActivityRegularizer/truediv_2:z:02^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp.^sequential_16/dense_16/BiasAdd/ReadVariableOp-^sequential_16/dense_16/MatMul/ReadVariableOp.^sequential_17/dense_17/BiasAdd/ReadVariableOp-^sequential_17/dense_17/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_16/dense_16/BiasAdd/ReadVariableOp-sequential_16/dense_16/BiasAdd/ReadVariableOp2\
,sequential_16/dense_16/MatMul/ReadVariableOp,sequential_16/dense_16/MatMul/ReadVariableOp2^
-sequential_17/dense_17/BiasAdd/ReadVariableOp-sequential_17/dense_17/BiasAdd/ReadVariableOp2\
,sequential_17/dense_17/MatMul/ReadVariableOp,sequential_17/dense_17/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580133

inputs$
dense_17_4580121:
??
dense_17_4580123:	?
identity?? dense_17/StatefulPartitionedCall?1dense_17/kernel/Regularizer/Square/ReadVariableOp?
 dense_17/StatefulPartitionedCallStatefulPartitionedCallinputsdense_17_4580121dense_17_4580123*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_45800772"
 dense_17/StatefulPartitionedCall?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_17_4580121* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_17/StatefulPartitionedCall2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580349
input_1)
sequential_16_4580324:
??$
sequential_16_4580326:	?)
sequential_17_4580330:
??$
sequential_17_4580332:	?
identity

identity_1??1dense_16/kernel/Regularizer/Square/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_16_4580324sequential_16_4580326*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_45799872'
%sequential_16/StatefulPartitionedCall?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_4580330sequential_17_4580332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_45801332'
%sequential_17/StatefulPartitionedCall?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_4580324* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_4580330* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:02^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity.sequential_16/StatefulPartitionedCall:output:12^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
%__inference_signature_wrapper_4580376
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_45798462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580750
dense_17_input;
'dense_17_matmul_readvariableop_resource:
??7
(dense_17_biasadd_readvariableop_resource:	?
identity??dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_17_input&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/BiasAdd}
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_17/Sigmoid?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentitydense_17/Sigmoid:y:0 ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_17_input
?
?
/__inference_sequential_17_layer_call_fn_4580682
dense_17_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_17_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_45801332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_17_input
?
?
__inference_loss_fn_0_4580787N
:dense_16_kernel_regularizer_square_readvariableop_resource:
??
identity??1dense_16/kernel/Regularizer/Square/ReadVariableOp?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_16_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentity#dense_16/kernel/Regularizer/mul:z:02^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp
?
?
E__inference_dense_17_layer_call_and_return_conditional_losses_4580810

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference__traced_restore_4580904
file_prefix4
 assignvariableop_dense_16_kernel:
??/
 assignvariableop_1_dense_16_bias:	?6
"assignvariableop_2_dense_17_kernel:
??/
 assignvariableop_3_dense_17_bias:	?

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
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

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_16_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_16_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_17_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_17_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

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
?"
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4579921

inputs$
dense_16_4579900:
??
dense_16_4579902:	?
identity

identity_1?? dense_16/StatefulPartitionedCall?1dense_16/kernel/Regularizer/Square/ReadVariableOp?
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_4579900dense_16_4579902*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_45798992"
 dense_16/StatefulPartitionedCall?
,dense_16/ActivityRegularizer/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *:
f5R3
1__inference_dense_16_activity_regularizer_45798752.
,dense_16/ActivityRegularizer/PartitionedCall?
"dense_16/ActivityRegularizer/ShapeShape)dense_16/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape?
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack?
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1?
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2?
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice?
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Cast?
$dense_16/ActivityRegularizer/truedivRealDiv5dense_16/ActivityRegularizer/PartitionedCall:output:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truediv?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_4579900* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity(dense_16/ActivityRegularizer/truediv:z:0!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580716

inputs;
'dense_17_matmul_readvariableop_resource:
??7
(dense_17_biasadd_readvariableop_resource:	?
identity??dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMulinputs&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/BiasAdd}
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_17/Sigmoid?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentitydense_17/Sigmoid:y:0 ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580211
x)
sequential_16_4580186:
??$
sequential_16_4580188:	?)
sequential_17_4580192:
??$
sequential_17_4580194:	?
identity

identity_1??1dense_16/kernel/Regularizer/Square/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallxsequential_16_4580186sequential_16_4580188*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_45799212'
%sequential_16/StatefulPartitionedCall?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_4580192sequential_17_4580194*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_45800902'
%sequential_17/StatefulPartitionedCall?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_4580186* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_4580192* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:02^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity.sequential_16/StatefulPartitionedCall:output:12^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?$
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580267
x)
sequential_16_4580242:
??$
sequential_16_4580244:	?)
sequential_17_4580248:
??$
sequential_17_4580250:	?
identity

identity_1??1dense_16/kernel/Regularizer/Square/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?%sequential_16/StatefulPartitionedCall?%sequential_17/StatefulPartitionedCall?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallxsequential_16_4580242sequential_16_4580244*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_16_layer_call_and_return_conditional_losses_45799872'
%sequential_16/StatefulPartitionedCall?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCall.sequential_16/StatefulPartitionedCall:output:0sequential_17_4580248sequential_17_4580250*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_45801332'
%sequential_17/StatefulPartitionedCall?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_4580242* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_4580248* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:02^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity.sequential_16/StatefulPartitionedCall:output:12^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp&^sequential_16/StatefulPartitionedCall&^sequential_17/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?"
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4579987

inputs$
dense_16_4579966:
??
dense_16_4579968:	?
identity

identity_1?? dense_16/StatefulPartitionedCall?1dense_16/kernel/Regularizer/Square/ReadVariableOp?
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_4579966dense_16_4579968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_45798992"
 dense_16/StatefulPartitionedCall?
,dense_16/ActivityRegularizer/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *:
f5R3
1__inference_dense_16_activity_regularizer_45798752.
,dense_16/ActivityRegularizer/PartitionedCall?
"dense_16/ActivityRegularizer/ShapeShape)dense_16/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape?
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack?
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1?
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2?
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice?
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Cast?
$dense_16/ActivityRegularizer/truedivRealDiv5dense_16/ActivityRegularizer/PartitionedCall:output:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2&
$dense_16/ActivityRegularizer/truediv?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_4579966* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentity)dense_16/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity(dense_16/ActivityRegularizer/truediv:z:0!^dense_16/StatefulPartitionedCall2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580733
dense_17_input;
'dense_17_matmul_readvariableop_resource:
??7
(dense_17_biasadd_readvariableop_resource:	?
identity??dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_17_input&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/BiasAdd}
dense_17/SigmoidSigmoiddense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_17/Sigmoid?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentitydense_17/Sigmoid:y:0 ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp:X T
(
_output_shapes
:??????????
(
_user_specified_namedense_17_input
?e
?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580522
xI
5sequential_16_dense_16_matmul_readvariableop_resource:
??E
6sequential_16_dense_16_biasadd_readvariableop_resource:	?I
5sequential_17_dense_17_matmul_readvariableop_resource:
??E
6sequential_17_dense_17_biasadd_readvariableop_resource:	?
identity

identity_1??1dense_16/kernel/Regularizer/Square/ReadVariableOp?1dense_17/kernel/Regularizer/Square/ReadVariableOp?-sequential_16/dense_16/BiasAdd/ReadVariableOp?,sequential_16/dense_16/MatMul/ReadVariableOp?-sequential_17/dense_17/BiasAdd/ReadVariableOp?,sequential_17/dense_17/MatMul/ReadVariableOp?
,sequential_16/dense_16/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_16/dense_16/MatMul/ReadVariableOp?
sequential_16/dense_16/MatMulMatMulx4sequential_16/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_16/dense_16/MatMul?
-sequential_16/dense_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_16/dense_16/BiasAdd/ReadVariableOp?
sequential_16/dense_16/BiasAddBiasAdd'sequential_16/dense_16/MatMul:product:05sequential_16/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_16/dense_16/BiasAdd?
sequential_16/dense_16/SigmoidSigmoid'sequential_16/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_16/dense_16/Sigmoid?
Asequential_16/dense_16/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asequential_16/dense_16/ActivityRegularizer/Mean/reduction_indices?
/sequential_16/dense_16/ActivityRegularizer/MeanMean"sequential_16/dense_16/Sigmoid:y:0Jsequential_16/dense_16/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?21
/sequential_16/dense_16/ActivityRegularizer/Mean?
4sequential_16/dense_16/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.26
4sequential_16/dense_16/ActivityRegularizer/Maximum/y?
2sequential_16/dense_16/ActivityRegularizer/MaximumMaximum8sequential_16/dense_16/ActivityRegularizer/Mean:output:0=sequential_16/dense_16/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?24
2sequential_16/dense_16/ActivityRegularizer/Maximum?
4sequential_16/dense_16/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<26
4sequential_16/dense_16/ActivityRegularizer/truediv/x?
2sequential_16/dense_16/ActivityRegularizer/truedivRealDiv=sequential_16/dense_16/ActivityRegularizer/truediv/x:output:06sequential_16/dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?24
2sequential_16/dense_16/ActivityRegularizer/truediv?
.sequential_16/dense_16/ActivityRegularizer/LogLog6sequential_16/dense_16/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?20
.sequential_16/dense_16/ActivityRegularizer/Log?
0sequential_16/dense_16/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<22
0sequential_16/dense_16/ActivityRegularizer/mul/x?
.sequential_16/dense_16/ActivityRegularizer/mulMul9sequential_16/dense_16/ActivityRegularizer/mul/x:output:02sequential_16/dense_16/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?20
.sequential_16/dense_16/ActivityRegularizer/mul?
0sequential_16/dense_16/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_16/dense_16/ActivityRegularizer/sub/x?
.sequential_16/dense_16/ActivityRegularizer/subSub9sequential_16/dense_16/ActivityRegularizer/sub/x:output:06sequential_16/dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?20
.sequential_16/dense_16/ActivityRegularizer/sub?
6sequential_16/dense_16/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?28
6sequential_16/dense_16/ActivityRegularizer/truediv_1/x?
4sequential_16/dense_16/ActivityRegularizer/truediv_1RealDiv?sequential_16/dense_16/ActivityRegularizer/truediv_1/x:output:02sequential_16/dense_16/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?26
4sequential_16/dense_16/ActivityRegularizer/truediv_1?
0sequential_16/dense_16/ActivityRegularizer/Log_1Log8sequential_16/dense_16/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?22
0sequential_16/dense_16/ActivityRegularizer/Log_1?
2sequential_16/dense_16/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?24
2sequential_16/dense_16/ActivityRegularizer/mul_1/x?
0sequential_16/dense_16/ActivityRegularizer/mul_1Mul;sequential_16/dense_16/ActivityRegularizer/mul_1/x:output:04sequential_16/dense_16/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?22
0sequential_16/dense_16/ActivityRegularizer/mul_1?
.sequential_16/dense_16/ActivityRegularizer/addAddV22sequential_16/dense_16/ActivityRegularizer/mul:z:04sequential_16/dense_16/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?20
.sequential_16/dense_16/ActivityRegularizer/add?
0sequential_16/dense_16/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_16/dense_16/ActivityRegularizer/Const?
.sequential_16/dense_16/ActivityRegularizer/SumSum2sequential_16/dense_16/ActivityRegularizer/add:z:09sequential_16/dense_16/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 20
.sequential_16/dense_16/ActivityRegularizer/Sum?
2sequential_16/dense_16/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_16/dense_16/ActivityRegularizer/mul_2/x?
0sequential_16/dense_16/ActivityRegularizer/mul_2Mul;sequential_16/dense_16/ActivityRegularizer/mul_2/x:output:07sequential_16/dense_16/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 22
0sequential_16/dense_16/ActivityRegularizer/mul_2?
0sequential_16/dense_16/ActivityRegularizer/ShapeShape"sequential_16/dense_16/Sigmoid:y:0*
T0*
_output_shapes
:22
0sequential_16/dense_16/ActivityRegularizer/Shape?
>sequential_16/dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_16/dense_16/ActivityRegularizer/strided_slice/stack?
@sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1?
@sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2?
8sequential_16/dense_16/ActivityRegularizer/strided_sliceStridedSlice9sequential_16/dense_16/ActivityRegularizer/Shape:output:0Gsequential_16/dense_16/ActivityRegularizer/strided_slice/stack:output:0Isequential_16/dense_16/ActivityRegularizer/strided_slice/stack_1:output:0Isequential_16/dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_16/dense_16/ActivityRegularizer/strided_slice?
/sequential_16/dense_16/ActivityRegularizer/CastCastAsequential_16/dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/sequential_16/dense_16/ActivityRegularizer/Cast?
4sequential_16/dense_16/ActivityRegularizer/truediv_2RealDiv4sequential_16/dense_16/ActivityRegularizer/mul_2:z:03sequential_16/dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 26
4sequential_16/dense_16/ActivityRegularizer/truediv_2?
,sequential_17/dense_17/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_17/dense_17/MatMul/ReadVariableOp?
sequential_17/dense_17/MatMulMatMul"sequential_16/dense_16/Sigmoid:y:04sequential_17/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_17/dense_17/MatMul?
-sequential_17/dense_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_17/dense_17/BiasAdd/ReadVariableOp?
sequential_17/dense_17/BiasAddBiasAdd'sequential_17/dense_17/MatMul:product:05sequential_17/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_17/BiasAdd?
sequential_17/dense_17/SigmoidSigmoid'sequential_17/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2 
sequential_17/dense_17/Sigmoid?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
1dense_17/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_17/kernel/Regularizer/Square/ReadVariableOp?
"dense_17/kernel/Regularizer/SquareSquare9dense_17/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_17/kernel/Regularizer/Square?
!dense_17/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_17/kernel/Regularizer/Const?
dense_17/kernel/Regularizer/SumSum&dense_17/kernel/Regularizer/Square:y:0*dense_17/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/Sum?
!dense_17/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_17/kernel/Regularizer/mul/x?
dense_17/kernel/Regularizer/mulMul*dense_17/kernel/Regularizer/mul/x:output:0(dense_17/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_17/kernel/Regularizer/mul?
IdentityIdentity"sequential_17/dense_17/Sigmoid:y:02^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp.^sequential_16/dense_16/BiasAdd/ReadVariableOp-^sequential_16/dense_16/MatMul/ReadVariableOp.^sequential_17/dense_17/BiasAdd/ReadVariableOp-^sequential_17/dense_17/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity8sequential_16/dense_16/ActivityRegularizer/truediv_2:z:02^dense_16/kernel/Regularizer/Square/ReadVariableOp2^dense_17/kernel/Regularizer/Square/ReadVariableOp.^sequential_16/dense_16/BiasAdd/ReadVariableOp-^sequential_16/dense_16/MatMul/ReadVariableOp.^sequential_17/dense_17/BiasAdd/ReadVariableOp-^sequential_17/dense_17/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2f
1dense_17/kernel/Regularizer/Square/ReadVariableOp1dense_17/kernel/Regularizer/Square/ReadVariableOp2^
-sequential_16/dense_16/BiasAdd/ReadVariableOp-sequential_16/dense_16/BiasAdd/ReadVariableOp2\
,sequential_16/dense_16/MatMul/ReadVariableOp,sequential_16/dense_16/MatMul/ReadVariableOp2^
-sequential_17/dense_17/BiasAdd/ReadVariableOp-sequential_17/dense_17/BiasAdd/ReadVariableOp2\
,sequential_17/dense_17/MatMul/ReadVariableOp,sequential_17/dense_17/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
E__inference_dense_16_layer_call_and_return_conditional_losses_4580847

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_16/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580640

inputs;
'dense_16_matmul_readvariableop_resource:
??7
(dense_16_biasadd_readvariableop_resource:	?
identity

identity_1??dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?1dense_16/kernel/Regularizer/Square/ReadVariableOp?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulinputs&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/BiasAdd}
dense_16/SigmoidSigmoiddense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_16/Sigmoid?
3dense_16/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 25
3dense_16/ActivityRegularizer/Mean/reduction_indices?
!dense_16/ActivityRegularizer/MeanMeandense_16/Sigmoid:y:0<dense_16/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2#
!dense_16/ActivityRegularizer/Mean?
&dense_16/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2(
&dense_16/ActivityRegularizer/Maximum/y?
$dense_16/ActivityRegularizer/MaximumMaximum*dense_16/ActivityRegularizer/Mean:output:0/dense_16/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2&
$dense_16/ActivityRegularizer/Maximum?
&dense_16/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2(
&dense_16/ActivityRegularizer/truediv/x?
$dense_16/ActivityRegularizer/truedivRealDiv/dense_16/ActivityRegularizer/truediv/x:output:0(dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2&
$dense_16/ActivityRegularizer/truediv?
 dense_16/ActivityRegularizer/LogLog(dense_16/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2"
 dense_16/ActivityRegularizer/Log?
"dense_16/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_16/ActivityRegularizer/mul/x?
 dense_16/ActivityRegularizer/mulMul+dense_16/ActivityRegularizer/mul/x:output:0$dense_16/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2"
 dense_16/ActivityRegularizer/mul?
"dense_16/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"dense_16/ActivityRegularizer/sub/x?
 dense_16/ActivityRegularizer/subSub+dense_16/ActivityRegularizer/sub/x:output:0(dense_16/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2"
 dense_16/ActivityRegularizer/sub?
(dense_16/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2*
(dense_16/ActivityRegularizer/truediv_1/x?
&dense_16/ActivityRegularizer/truediv_1RealDiv1dense_16/ActivityRegularizer/truediv_1/x:output:0$dense_16/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2(
&dense_16/ActivityRegularizer/truediv_1?
"dense_16/ActivityRegularizer/Log_1Log*dense_16/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2$
"dense_16/ActivityRegularizer/Log_1?
$dense_16/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2&
$dense_16/ActivityRegularizer/mul_1/x?
"dense_16/ActivityRegularizer/mul_1Mul-dense_16/ActivityRegularizer/mul_1/x:output:0&dense_16/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2$
"dense_16/ActivityRegularizer/mul_1?
 dense_16/ActivityRegularizer/addAddV2$dense_16/ActivityRegularizer/mul:z:0&dense_16/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2"
 dense_16/ActivityRegularizer/add?
"dense_16/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"dense_16/ActivityRegularizer/Const?
 dense_16/ActivityRegularizer/SumSum$dense_16/ActivityRegularizer/add:z:0+dense_16/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_16/ActivityRegularizer/Sum?
$dense_16/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$dense_16/ActivityRegularizer/mul_2/x?
"dense_16/ActivityRegularizer/mul_2Mul-dense_16/ActivityRegularizer/mul_2/x:output:0)dense_16/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2$
"dense_16/ActivityRegularizer/mul_2?
"dense_16/ActivityRegularizer/ShapeShapedense_16/Sigmoid:y:0*
T0*
_output_shapes
:2$
"dense_16/ActivityRegularizer/Shape?
0dense_16/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0dense_16/ActivityRegularizer/strided_slice/stack?
2dense_16/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_1?
2dense_16/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2dense_16/ActivityRegularizer/strided_slice/stack_2?
*dense_16/ActivityRegularizer/strided_sliceStridedSlice+dense_16/ActivityRegularizer/Shape:output:09dense_16/ActivityRegularizer/strided_slice/stack:output:0;dense_16/ActivityRegularizer/strided_slice/stack_1:output:0;dense_16/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*dense_16/ActivityRegularizer/strided_slice?
!dense_16/ActivityRegularizer/CastCast3dense_16/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!dense_16/ActivityRegularizer/Cast?
&dense_16/ActivityRegularizer/truediv_2RealDiv&dense_16/ActivityRegularizer/mul_2:z:0%dense_16/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2(
&dense_16/ActivityRegularizer/truediv_2?
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp?
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2$
"dense_16/kernel/Regularizer/Square?
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const?
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum?
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_16/kernel/Regularizer/mul/x?
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul?
IdentityIdentitydense_16/Sigmoid:y:0 ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity*dense_16/ActivityRegularizer/truediv_2:z:0 ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Q
1__inference_dense_16_activity_regularizer_4579875

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
 *???.2
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
?#<2
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
?#<2
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
 *  ??2
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
 *?p}?2
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
 *?p}?2	
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
:?????????2
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
 *  ??2	
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
activation"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????=
output_11
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
?
history
encoder
decoder
	variables
trainable_variables
regularization_losses
	keras_api

signatures
8_default_save_signature
9__call__
*:&call_and_return_all_conditional_losses"?
_tf_keras_model?{"name": "autoencoder_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
 "
trackable_dict_wrapper
?
	layer_with_weights-0
	layer-0

	variables
trainable_variables
regularization_losses
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300]}, "float32", "input_9"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_17_input"}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 300, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [300, 128]}, "float32", "dense_17_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_17_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 300, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
?
non_trainable_variables
layer_regularization_losses
	variables
metrics
trainable_variables

layers
layer_metrics
regularization_losses
9__call__
8_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"?

_tf_keras_layer?
{"name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
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
?
 non_trainable_variables
!layer_regularization_losses

	variables
"metrics
trainable_variables

#layers
$layer_metrics
regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?	

kernel
bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
*C&call_and_return_all_conditional_losses
D__call__"?
_tf_keras_layer?{"name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 300, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [300, 128]}}
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
?
)non_trainable_variables
*layer_regularization_losses
	variables
+metrics
trainable_variables

,layers
-layer_metrics
regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_16/kernel
:?2dense_16/bias
#:!
??2dense_17/kernel
:?2dense_17/bias
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
?
.layer_regularization_losses
	variables
/metrics
trainable_variables
0layer_metrics

1layers
2non_trainable_variables
regularization_losses
A__call__
Factivity_regularizer_fn
*@&call_and_return_all_conditional_losses
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
?
3layer_regularization_losses
%	variables
4metrics
&trainable_variables
5layer_metrics

6layers
7non_trainable_variables
'regularization_losses
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
"__inference__wrapped_model_4579846?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
/__inference_autoencoder_8_layer_call_fn_4580223
/__inference_autoencoder_8_layer_call_fn_4580390
/__inference_autoencoder_8_layer_call_fn_4580404
/__inference_autoencoder_8_layer_call_fn_4580293?
???
FullArgSpec$
args?
jself
jX

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580463
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580522
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580321
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580349?
???
FullArgSpec$
args?
jself
jX

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_sequential_16_layer_call_fn_4579929
/__inference_sequential_16_layer_call_fn_4580538
/__inference_sequential_16_layer_call_fn_4580548
/__inference_sequential_16_layer_call_fn_4580005?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580594
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580640
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580029
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580053?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_sequential_17_layer_call_fn_4580655
/__inference_sequential_17_layer_call_fn_4580664
/__inference_sequential_17_layer_call_fn_4580673
/__inference_sequential_17_layer_call_fn_4580682?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580699
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580716
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580733
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580750?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_signature_wrapper_4580376input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_dense_16_layer_call_and_return_all_conditional_losses_4580767?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_16_layer_call_fn_4580776?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_4580787?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
E__inference_dense_17_layer_call_and_return_conditional_losses_4580810?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_17_layer_call_fn_4580819?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_1_4580830?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
1__inference_dense_16_activity_regularizer_4579875?
???
FullArgSpec!
args?
jself
j
activation
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
E__inference_dense_16_layer_call_and_return_conditional_losses_4580847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_4579846o1?.
'?$
"?
input_1??????????
? "4?1
/
output_1#? 
output_1???????????
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580321s5?2
+?(
"?
input_1??????????
p 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580349s5?2
+?(
"?
input_1??????????
p
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580463m/?,
%?"
?
X??????????
p 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_autoencoder_8_layer_call_and_return_conditional_losses_4580522m/?,
%?"
?
X??????????
p
? "4?1
?
0??????????
?
?	
1/0 ?
/__inference_autoencoder_8_layer_call_fn_4580223X5?2
+?(
"?
input_1??????????
p 
? "????????????
/__inference_autoencoder_8_layer_call_fn_4580293X5?2
+?(
"?
input_1??????????
p
? "????????????
/__inference_autoencoder_8_layer_call_fn_4580390R/?,
%?"
?
X??????????
p 
? "????????????
/__inference_autoencoder_8_layer_call_fn_4580404R/?,
%?"
?
X??????????
p
? "???????????d
1__inference_dense_16_activity_regularizer_4579875/$?!
?
?

activation
? "? ?
I__inference_dense_16_layer_call_and_return_all_conditional_losses_4580767l0?-
&?#
!?
inputs??????????
? "4?1
?
0??????????
?
?	
1/0 ?
E__inference_dense_16_layer_call_and_return_conditional_losses_4580847^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_16_layer_call_fn_4580776Q0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_17_layer_call_and_return_conditional_losses_4580810^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_17_layer_call_fn_4580819Q0?-
&?#
!?
inputs??????????
? "???????????<
__inference_loss_fn_0_4580787?

? 
? "? <
__inference_loss_fn_1_4580830?

? 
? "? ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580029u9?6
/?,
"?
input_9??????????
p 

 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580053u9?6
/?,
"?
input_9??????????
p

 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580594t8?5
.?+
!?
inputs??????????
p 

 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_4580640t8?5
.?+
!?
inputs??????????
p

 
? "4?1
?
0??????????
?
?	
1/0 ?
/__inference_sequential_16_layer_call_fn_4579929Z9?6
/?,
"?
input_9??????????
p 

 
? "????????????
/__inference_sequential_16_layer_call_fn_4580005Z9?6
/?,
"?
input_9??????????
p

 
? "????????????
/__inference_sequential_16_layer_call_fn_4580538Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
/__inference_sequential_16_layer_call_fn_4580548Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580699f8?5
.?+
!?
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580716f8?5
.?+
!?
inputs??????????
p

 
? "&?#
?
0??????????
? ?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580733n@?=
6?3
)?&
dense_17_input??????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_17_layer_call_and_return_conditional_losses_4580750n@?=
6?3
)?&
dense_17_input??????????
p

 
? "&?#
?
0??????????
? ?
/__inference_sequential_17_layer_call_fn_4580655a@?=
6?3
)?&
dense_17_input??????????
p 

 
? "????????????
/__inference_sequential_17_layer_call_fn_4580664Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
/__inference_sequential_17_layer_call_fn_4580673Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
/__inference_sequential_17_layer_call_fn_4580682a@?=
6?3
)?&
dense_17_input??????????
p

 
? "????????????
%__inference_signature_wrapper_4580376z<?9
? 
2?/
-
input_1"?
input_1??????????"4?1
/
output_1#? 
output_1??????????