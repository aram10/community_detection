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
dense_138/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ *!
shared_namedense_138/kernel
u
$dense_138/kernel/Read/ReadVariableOpReadVariableOpdense_138/kernel*
_output_shapes

:^ *
dtype0
t
dense_138/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_138/bias
m
"dense_138/bias/Read/ReadVariableOpReadVariableOpdense_138/bias*
_output_shapes
: *
dtype0
|
dense_139/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^*!
shared_namedense_139/kernel
u
$dense_139/kernel/Read/ReadVariableOpReadVariableOpdense_139/kernel*
_output_shapes

: ^*
dtype0
t
dense_139/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_139/bias
m
"dense_139/bias/Read/ReadVariableOpReadVariableOpdense_139/bias*
_output_shapes
:^*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
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
?
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
?
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
?
)layer_regularization_losses
*non_trainable_variables
+metrics
trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
VT
VARIABLE_VALUEdense_138/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_138/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_139/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_139/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
?
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
:?????????^*
dtype0*
shape:?????????^
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_138/kerneldense_138/biasdense_139/kerneldense_139/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_16662536
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_138/kernel/Read/ReadVariableOp"dense_138/bias/Read/ReadVariableOp$dense_139/kernel/Read/ReadVariableOp"dense_139/bias/Read/ReadVariableOpConst*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_16663042
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_138/kerneldense_138/biasdense_139/kerneldense_139/bias*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_16663064??	
?
?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662910
dense_139_input:
(dense_139_matmul_readvariableop_resource: ^7
)dense_139_biasadd_readvariableop_resource:^
identity?? dense_139/BiasAdd/ReadVariableOp?dense_139/MatMul/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?
dense_139/MatMul/ReadVariableOpReadVariableOp(dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_139/MatMul/ReadVariableOp?
dense_139/MatMulMatMuldense_139_input'dense_139/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_139/MatMul?
 dense_139/BiasAdd/ReadVariableOpReadVariableOp)dense_139_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_139/BiasAdd/ReadVariableOp?
dense_139/BiasAddBiasAdddense_139/MatMul:product:0(dense_139/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_139/BiasAdd
dense_139/SigmoidSigmoiddense_139/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_139/Sigmoid?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentitydense_139/Sigmoid:y:0!^dense_139/BiasAdd/ReadVariableOp ^dense_139/MatMul/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_139/BiasAdd/ReadVariableOp dense_139/BiasAdd/ReadVariableOp2B
dense_139/MatMul/ReadVariableOpdense_139/MatMul/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_139_input
?
?
G__inference_dense_138_layer_call_and_return_conditional_losses_16663007

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_138/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662250

inputs$
dense_139_16662238: ^ 
dense_139_16662240:^
identity??!dense_139/StatefulPartitionedCall?2dense_139/kernel/Regularizer/Square/ReadVariableOp?
!dense_139/StatefulPartitionedCallStatefulPartitionedCallinputsdense_139_16662238dense_139_16662240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_139_layer_call_and_return_conditional_losses_166622372#
!dense_139/StatefulPartitionedCall?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_139_16662238*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity*dense_139/StatefulPartitionedCall:output:0"^dense_139/StatefulPartitionedCall3^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_139/StatefulPartitionedCall!dense_139/StatefulPartitionedCall2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662293

inputs$
dense_139_16662281: ^ 
dense_139_16662283:^
identity??!dense_139/StatefulPartitionedCall?2dense_139/kernel/Regularizer/Square/ReadVariableOp?
!dense_139/StatefulPartitionedCallStatefulPartitionedCallinputsdense_139_16662281dense_139_16662283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_139_layer_call_and_return_conditional_losses_166622372#
!dense_139/StatefulPartitionedCall?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_139_16662281*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity*dense_139/StatefulPartitionedCall:output:0"^dense_139/StatefulPartitionedCall3^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_139/StatefulPartitionedCall!dense_139/StatefulPartitionedCall2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_sequential_138_layer_call_fn_16662089
input_70
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_70unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_138_layer_call_and_return_conditional_losses_166620812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_70
?%
?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662427
x)
sequential_138_16662402:^ %
sequential_138_16662404: )
sequential_139_16662408: ^%
sequential_139_16662410:^
identity

identity_1??2dense_138/kernel/Regularizer/Square/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?&sequential_138/StatefulPartitionedCall?&sequential_139/StatefulPartitionedCall?
&sequential_138/StatefulPartitionedCallStatefulPartitionedCallxsequential_138_16662402sequential_138_16662404*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_138_layer_call_and_return_conditional_losses_166621472(
&sequential_138/StatefulPartitionedCall?
&sequential_139/StatefulPartitionedCallStatefulPartitionedCall/sequential_138/StatefulPartitionedCall:output:0sequential_139_16662408sequential_139_16662410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_139_layer_call_and_return_conditional_losses_166622932(
&sequential_139/StatefulPartitionedCall?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_138_16662402*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_139_16662408*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity/sequential_139/StatefulPartitionedCall:output:03^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp'^sequential_138/StatefulPartitionedCall'^sequential_139/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_138/StatefulPartitionedCall:output:13^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp'^sequential_138/StatefulPartitionedCall'^sequential_139/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_138/StatefulPartitionedCall&sequential_138/StatefulPartitionedCall2P
&sequential_139/StatefulPartitionedCall&sequential_139/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
1__inference_sequential_139_layer_call_fn_16662833

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_139_layer_call_and_return_conditional_losses_166622932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
S
3__inference_dense_138_activity_regularizer_16662035

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
activation
?
?
1__inference_autoencoder_69_layer_call_fn_16662383
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_166623712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
K__inference_dense_138_layer_call_and_return_all_conditional_losses_16662936

inputs
unknown:^ 
	unknown_0: 
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_138_layer_call_and_return_conditional_losses_166620592
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
GPU 2J 8? *<
f7R5
3__inference_dense_138_activity_regularizer_166620352
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

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
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?#
?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662147

inputs$
dense_138_16662126:^  
dense_138_16662128: 
identity

identity_1??!dense_138/StatefulPartitionedCall?2dense_138/kernel/Regularizer/Square/ReadVariableOp?
!dense_138/StatefulPartitionedCallStatefulPartitionedCallinputsdense_138_16662126dense_138_16662128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_138_layer_call_and_return_conditional_losses_166620592#
!dense_138/StatefulPartitionedCall?
-dense_138/ActivityRegularizer/PartitionedCallPartitionedCall*dense_138/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *<
f7R5
3__inference_dense_138_activity_regularizer_166620352/
-dense_138/ActivityRegularizer/PartitionedCall?
#dense_138/ActivityRegularizer/ShapeShape*dense_138/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_138/ActivityRegularizer/Shape?
1dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_138/ActivityRegularizer/strided_slice/stack?
3dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_1?
3dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_2?
+dense_138/ActivityRegularizer/strided_sliceStridedSlice,dense_138/ActivityRegularizer/Shape:output:0:dense_138/ActivityRegularizer/strided_slice/stack:output:0<dense_138/ActivityRegularizer/strided_slice/stack_1:output:0<dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_138/ActivityRegularizer/strided_slice?
"dense_138/ActivityRegularizer/CastCast4dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_138/ActivityRegularizer/Cast?
%dense_138/ActivityRegularizer/truedivRealDiv6dense_138/ActivityRegularizer/PartitionedCall:output:0&dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_138/ActivityRegularizer/truediv?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_138_16662126*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentity*dense_138/StatefulPartitionedCall:output:0"^dense_138/StatefulPartitionedCall3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_138/ActivityRegularizer/truediv:z:0"^dense_138/StatefulPartitionedCall3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_138/StatefulPartitionedCall!dense_138/StatefulPartitionedCall2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
G__inference_dense_139_layer_call_and_return_conditional_losses_16662979

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????^2	
Sigmoid?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_dense_138_layer_call_and_return_conditional_losses_16662059

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_138/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?%
?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662509
input_1)
sequential_138_16662484:^ %
sequential_138_16662486: )
sequential_139_16662490: ^%
sequential_139_16662492:^
identity

identity_1??2dense_138/kernel/Regularizer/Square/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?&sequential_138/StatefulPartitionedCall?&sequential_139/StatefulPartitionedCall?
&sequential_138/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_138_16662484sequential_138_16662486*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_138_layer_call_and_return_conditional_losses_166621472(
&sequential_138/StatefulPartitionedCall?
&sequential_139/StatefulPartitionedCallStatefulPartitionedCall/sequential_138/StatefulPartitionedCall:output:0sequential_139_16662490sequential_139_16662492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_139_layer_call_and_return_conditional_losses_166622932(
&sequential_139/StatefulPartitionedCall?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_138_16662484*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_139_16662490*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity/sequential_139/StatefulPartitionedCall:output:03^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp'^sequential_138/StatefulPartitionedCall'^sequential_139/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_138/StatefulPartitionedCall:output:13^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp'^sequential_138/StatefulPartitionedCall'^sequential_139/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_138/StatefulPartitionedCall&sequential_138/StatefulPartitionedCall2P
&sequential_139/StatefulPartitionedCall&sequential_139/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
__inference_loss_fn_0_16662947M
;dense_138_kernel_regularizer_square_readvariableop_resource:^ 
identity??2dense_138/kernel/Regularizer/Square/ReadVariableOp?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_138_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentity$dense_138/kernel/Regularizer/mul:z:03^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp
?B
?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662800

inputs:
(dense_138_matmul_readvariableop_resource:^ 7
)dense_138_biasadd_readvariableop_resource: 
identity

identity_1?? dense_138/BiasAdd/ReadVariableOp?dense_138/MatMul/ReadVariableOp?2dense_138/kernel/Regularizer/Square/ReadVariableOp?
dense_138/MatMul/ReadVariableOpReadVariableOp(dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_138/MatMul/ReadVariableOp?
dense_138/MatMulMatMulinputs'dense_138/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_138/MatMul?
 dense_138/BiasAdd/ReadVariableOpReadVariableOp)dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_138/BiasAdd/ReadVariableOp?
dense_138/BiasAddBiasAdddense_138/MatMul:product:0(dense_138/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_138/BiasAdd
dense_138/SigmoidSigmoiddense_138/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_138/Sigmoid?
4dense_138/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_138/ActivityRegularizer/Mean/reduction_indices?
"dense_138/ActivityRegularizer/MeanMeandense_138/Sigmoid:y:0=dense_138/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_138/ActivityRegularizer/Mean?
'dense_138/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_138/ActivityRegularizer/Maximum/y?
%dense_138/ActivityRegularizer/MaximumMaximum+dense_138/ActivityRegularizer/Mean:output:00dense_138/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_138/ActivityRegularizer/Maximum?
'dense_138/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_138/ActivityRegularizer/truediv/x?
%dense_138/ActivityRegularizer/truedivRealDiv0dense_138/ActivityRegularizer/truediv/x:output:0)dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_138/ActivityRegularizer/truediv?
!dense_138/ActivityRegularizer/LogLog)dense_138/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/Log?
#dense_138/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_138/ActivityRegularizer/mul/x?
!dense_138/ActivityRegularizer/mulMul,dense_138/ActivityRegularizer/mul/x:output:0%dense_138/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/mul?
#dense_138/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_138/ActivityRegularizer/sub/x?
!dense_138/ActivityRegularizer/subSub,dense_138/ActivityRegularizer/sub/x:output:0)dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/sub?
)dense_138/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_138/ActivityRegularizer/truediv_1/x?
'dense_138/ActivityRegularizer/truediv_1RealDiv2dense_138/ActivityRegularizer/truediv_1/x:output:0%dense_138/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_138/ActivityRegularizer/truediv_1?
#dense_138/ActivityRegularizer/Log_1Log+dense_138/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_138/ActivityRegularizer/Log_1?
%dense_138/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_138/ActivityRegularizer/mul_1/x?
#dense_138/ActivityRegularizer/mul_1Mul.dense_138/ActivityRegularizer/mul_1/x:output:0'dense_138/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_138/ActivityRegularizer/mul_1?
!dense_138/ActivityRegularizer/addAddV2%dense_138/ActivityRegularizer/mul:z:0'dense_138/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/add?
#dense_138/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_138/ActivityRegularizer/Const?
!dense_138/ActivityRegularizer/SumSum%dense_138/ActivityRegularizer/add:z:0,dense_138/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/Sum?
%dense_138/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_138/ActivityRegularizer/mul_2/x?
#dense_138/ActivityRegularizer/mul_2Mul.dense_138/ActivityRegularizer/mul_2/x:output:0*dense_138/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_138/ActivityRegularizer/mul_2?
#dense_138/ActivityRegularizer/ShapeShapedense_138/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_138/ActivityRegularizer/Shape?
1dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_138/ActivityRegularizer/strided_slice/stack?
3dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_1?
3dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_2?
+dense_138/ActivityRegularizer/strided_sliceStridedSlice,dense_138/ActivityRegularizer/Shape:output:0:dense_138/ActivityRegularizer/strided_slice/stack:output:0<dense_138/ActivityRegularizer/strided_slice/stack_1:output:0<dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_138/ActivityRegularizer/strided_slice?
"dense_138/ActivityRegularizer/CastCast4dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_138/ActivityRegularizer/Cast?
'dense_138/ActivityRegularizer/truediv_2RealDiv'dense_138/ActivityRegularizer/mul_2:z:0&dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_138/ActivityRegularizer/truediv_2?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentitydense_138/Sigmoid:y:0!^dense_138/BiasAdd/ReadVariableOp ^dense_138/MatMul/ReadVariableOp3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_138/ActivityRegularizer/truediv_2:z:0!^dense_138/BiasAdd/ReadVariableOp ^dense_138/MatMul/ReadVariableOp3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_138/BiasAdd/ReadVariableOp dense_138/BiasAdd/ReadVariableOp2B
dense_138/MatMul/ReadVariableOpdense_138/MatMul/ReadVariableOp2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_sequential_138_layer_call_fn_16662698

inputs
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_138_layer_call_and_return_conditional_losses_166620812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_sequential_138_layer_call_fn_16662708

inputs
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_138_layer_call_and_return_conditional_losses_166621472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?h
?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662682
xI
7sequential_138_dense_138_matmul_readvariableop_resource:^ F
8sequential_138_dense_138_biasadd_readvariableop_resource: I
7sequential_139_dense_139_matmul_readvariableop_resource: ^F
8sequential_139_dense_139_biasadd_readvariableop_resource:^
identity

identity_1??2dense_138/kernel/Regularizer/Square/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?/sequential_138/dense_138/BiasAdd/ReadVariableOp?.sequential_138/dense_138/MatMul/ReadVariableOp?/sequential_139/dense_139/BiasAdd/ReadVariableOp?.sequential_139/dense_139/MatMul/ReadVariableOp?
.sequential_138/dense_138/MatMul/ReadVariableOpReadVariableOp7sequential_138_dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_138/dense_138/MatMul/ReadVariableOp?
sequential_138/dense_138/MatMulMatMulx6sequential_138/dense_138/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_138/dense_138/MatMul?
/sequential_138/dense_138/BiasAdd/ReadVariableOpReadVariableOp8sequential_138_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_138/dense_138/BiasAdd/ReadVariableOp?
 sequential_138/dense_138/BiasAddBiasAdd)sequential_138/dense_138/MatMul:product:07sequential_138/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_138/dense_138/BiasAdd?
 sequential_138/dense_138/SigmoidSigmoid)sequential_138/dense_138/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_138/dense_138/Sigmoid?
Csequential_138/dense_138/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_138/dense_138/ActivityRegularizer/Mean/reduction_indices?
1sequential_138/dense_138/ActivityRegularizer/MeanMean$sequential_138/dense_138/Sigmoid:y:0Lsequential_138/dense_138/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_138/dense_138/ActivityRegularizer/Mean?
6sequential_138/dense_138/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_138/dense_138/ActivityRegularizer/Maximum/y?
4sequential_138/dense_138/ActivityRegularizer/MaximumMaximum:sequential_138/dense_138/ActivityRegularizer/Mean:output:0?sequential_138/dense_138/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_138/dense_138/ActivityRegularizer/Maximum?
6sequential_138/dense_138/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_138/dense_138/ActivityRegularizer/truediv/x?
4sequential_138/dense_138/ActivityRegularizer/truedivRealDiv?sequential_138/dense_138/ActivityRegularizer/truediv/x:output:08sequential_138/dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_138/dense_138/ActivityRegularizer/truediv?
0sequential_138/dense_138/ActivityRegularizer/LogLog8sequential_138/dense_138/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/Log?
2sequential_138/dense_138/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_138/dense_138/ActivityRegularizer/mul/x?
0sequential_138/dense_138/ActivityRegularizer/mulMul;sequential_138/dense_138/ActivityRegularizer/mul/x:output:04sequential_138/dense_138/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/mul?
2sequential_138/dense_138/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_138/dense_138/ActivityRegularizer/sub/x?
0sequential_138/dense_138/ActivityRegularizer/subSub;sequential_138/dense_138/ActivityRegularizer/sub/x:output:08sequential_138/dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/sub?
8sequential_138/dense_138/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_138/dense_138/ActivityRegularizer/truediv_1/x?
6sequential_138/dense_138/ActivityRegularizer/truediv_1RealDivAsequential_138/dense_138/ActivityRegularizer/truediv_1/x:output:04sequential_138/dense_138/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_138/dense_138/ActivityRegularizer/truediv_1?
2sequential_138/dense_138/ActivityRegularizer/Log_1Log:sequential_138/dense_138/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_138/dense_138/ActivityRegularizer/Log_1?
4sequential_138/dense_138/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_138/dense_138/ActivityRegularizer/mul_1/x?
2sequential_138/dense_138/ActivityRegularizer/mul_1Mul=sequential_138/dense_138/ActivityRegularizer/mul_1/x:output:06sequential_138/dense_138/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_138/dense_138/ActivityRegularizer/mul_1?
0sequential_138/dense_138/ActivityRegularizer/addAddV24sequential_138/dense_138/ActivityRegularizer/mul:z:06sequential_138/dense_138/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/add?
2sequential_138/dense_138/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_138/dense_138/ActivityRegularizer/Const?
0sequential_138/dense_138/ActivityRegularizer/SumSum4sequential_138/dense_138/ActivityRegularizer/add:z:0;sequential_138/dense_138/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/Sum?
4sequential_138/dense_138/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_138/dense_138/ActivityRegularizer/mul_2/x?
2sequential_138/dense_138/ActivityRegularizer/mul_2Mul=sequential_138/dense_138/ActivityRegularizer/mul_2/x:output:09sequential_138/dense_138/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_138/dense_138/ActivityRegularizer/mul_2?
2sequential_138/dense_138/ActivityRegularizer/ShapeShape$sequential_138/dense_138/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_138/dense_138/ActivityRegularizer/Shape?
@sequential_138/dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_138/dense_138/ActivityRegularizer/strided_slice/stack?
Bsequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1?
Bsequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2?
:sequential_138/dense_138/ActivityRegularizer/strided_sliceStridedSlice;sequential_138/dense_138/ActivityRegularizer/Shape:output:0Isequential_138/dense_138/ActivityRegularizer/strided_slice/stack:output:0Ksequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_138/dense_138/ActivityRegularizer/strided_slice?
1sequential_138/dense_138/ActivityRegularizer/CastCastCsequential_138/dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_138/dense_138/ActivityRegularizer/Cast?
6sequential_138/dense_138/ActivityRegularizer/truediv_2RealDiv6sequential_138/dense_138/ActivityRegularizer/mul_2:z:05sequential_138/dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_138/dense_138/ActivityRegularizer/truediv_2?
.sequential_139/dense_139/MatMul/ReadVariableOpReadVariableOp7sequential_139_dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_139/dense_139/MatMul/ReadVariableOp?
sequential_139/dense_139/MatMulMatMul$sequential_138/dense_138/Sigmoid:y:06sequential_139/dense_139/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_139/dense_139/MatMul?
/sequential_139/dense_139/BiasAdd/ReadVariableOpReadVariableOp8sequential_139_dense_139_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_139/dense_139/BiasAdd/ReadVariableOp?
 sequential_139/dense_139/BiasAddBiasAdd)sequential_139/dense_139/MatMul:product:07sequential_139/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_139/dense_139/BiasAdd?
 sequential_139/dense_139/SigmoidSigmoid)sequential_139/dense_139/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_139/dense_139/Sigmoid?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_138_dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_139_dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity$sequential_139/dense_139/Sigmoid:y:03^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp0^sequential_138/dense_138/BiasAdd/ReadVariableOp/^sequential_138/dense_138/MatMul/ReadVariableOp0^sequential_139/dense_139/BiasAdd/ReadVariableOp/^sequential_139/dense_139/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_138/dense_138/ActivityRegularizer/truediv_2:z:03^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp0^sequential_138/dense_138/BiasAdd/ReadVariableOp/^sequential_138/dense_138/MatMul/ReadVariableOp0^sequential_139/dense_139/BiasAdd/ReadVariableOp/^sequential_139/dense_139/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_138/dense_138/BiasAdd/ReadVariableOp/sequential_138/dense_138/BiasAdd/ReadVariableOp2`
.sequential_138/dense_138/MatMul/ReadVariableOp.sequential_138/dense_138/MatMul/ReadVariableOp2b
/sequential_139/dense_139/BiasAdd/ReadVariableOp/sequential_139/dense_139/BiasAdd/ReadVariableOp2`
.sequential_139/dense_139/MatMul/ReadVariableOp.sequential_139/dense_139/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?h
?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662623
xI
7sequential_138_dense_138_matmul_readvariableop_resource:^ F
8sequential_138_dense_138_biasadd_readvariableop_resource: I
7sequential_139_dense_139_matmul_readvariableop_resource: ^F
8sequential_139_dense_139_biasadd_readvariableop_resource:^
identity

identity_1??2dense_138/kernel/Regularizer/Square/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?/sequential_138/dense_138/BiasAdd/ReadVariableOp?.sequential_138/dense_138/MatMul/ReadVariableOp?/sequential_139/dense_139/BiasAdd/ReadVariableOp?.sequential_139/dense_139/MatMul/ReadVariableOp?
.sequential_138/dense_138/MatMul/ReadVariableOpReadVariableOp7sequential_138_dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_138/dense_138/MatMul/ReadVariableOp?
sequential_138/dense_138/MatMulMatMulx6sequential_138/dense_138/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_138/dense_138/MatMul?
/sequential_138/dense_138/BiasAdd/ReadVariableOpReadVariableOp8sequential_138_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_138/dense_138/BiasAdd/ReadVariableOp?
 sequential_138/dense_138/BiasAddBiasAdd)sequential_138/dense_138/MatMul:product:07sequential_138/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_138/dense_138/BiasAdd?
 sequential_138/dense_138/SigmoidSigmoid)sequential_138/dense_138/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_138/dense_138/Sigmoid?
Csequential_138/dense_138/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_138/dense_138/ActivityRegularizer/Mean/reduction_indices?
1sequential_138/dense_138/ActivityRegularizer/MeanMean$sequential_138/dense_138/Sigmoid:y:0Lsequential_138/dense_138/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_138/dense_138/ActivityRegularizer/Mean?
6sequential_138/dense_138/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_138/dense_138/ActivityRegularizer/Maximum/y?
4sequential_138/dense_138/ActivityRegularizer/MaximumMaximum:sequential_138/dense_138/ActivityRegularizer/Mean:output:0?sequential_138/dense_138/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_138/dense_138/ActivityRegularizer/Maximum?
6sequential_138/dense_138/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_138/dense_138/ActivityRegularizer/truediv/x?
4sequential_138/dense_138/ActivityRegularizer/truedivRealDiv?sequential_138/dense_138/ActivityRegularizer/truediv/x:output:08sequential_138/dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_138/dense_138/ActivityRegularizer/truediv?
0sequential_138/dense_138/ActivityRegularizer/LogLog8sequential_138/dense_138/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/Log?
2sequential_138/dense_138/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_138/dense_138/ActivityRegularizer/mul/x?
0sequential_138/dense_138/ActivityRegularizer/mulMul;sequential_138/dense_138/ActivityRegularizer/mul/x:output:04sequential_138/dense_138/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/mul?
2sequential_138/dense_138/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_138/dense_138/ActivityRegularizer/sub/x?
0sequential_138/dense_138/ActivityRegularizer/subSub;sequential_138/dense_138/ActivityRegularizer/sub/x:output:08sequential_138/dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/sub?
8sequential_138/dense_138/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_138/dense_138/ActivityRegularizer/truediv_1/x?
6sequential_138/dense_138/ActivityRegularizer/truediv_1RealDivAsequential_138/dense_138/ActivityRegularizer/truediv_1/x:output:04sequential_138/dense_138/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_138/dense_138/ActivityRegularizer/truediv_1?
2sequential_138/dense_138/ActivityRegularizer/Log_1Log:sequential_138/dense_138/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_138/dense_138/ActivityRegularizer/Log_1?
4sequential_138/dense_138/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_138/dense_138/ActivityRegularizer/mul_1/x?
2sequential_138/dense_138/ActivityRegularizer/mul_1Mul=sequential_138/dense_138/ActivityRegularizer/mul_1/x:output:06sequential_138/dense_138/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_138/dense_138/ActivityRegularizer/mul_1?
0sequential_138/dense_138/ActivityRegularizer/addAddV24sequential_138/dense_138/ActivityRegularizer/mul:z:06sequential_138/dense_138/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/add?
2sequential_138/dense_138/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_138/dense_138/ActivityRegularizer/Const?
0sequential_138/dense_138/ActivityRegularizer/SumSum4sequential_138/dense_138/ActivityRegularizer/add:z:0;sequential_138/dense_138/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_138/dense_138/ActivityRegularizer/Sum?
4sequential_138/dense_138/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_138/dense_138/ActivityRegularizer/mul_2/x?
2sequential_138/dense_138/ActivityRegularizer/mul_2Mul=sequential_138/dense_138/ActivityRegularizer/mul_2/x:output:09sequential_138/dense_138/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_138/dense_138/ActivityRegularizer/mul_2?
2sequential_138/dense_138/ActivityRegularizer/ShapeShape$sequential_138/dense_138/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_138/dense_138/ActivityRegularizer/Shape?
@sequential_138/dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_138/dense_138/ActivityRegularizer/strided_slice/stack?
Bsequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1?
Bsequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2?
:sequential_138/dense_138/ActivityRegularizer/strided_sliceStridedSlice;sequential_138/dense_138/ActivityRegularizer/Shape:output:0Isequential_138/dense_138/ActivityRegularizer/strided_slice/stack:output:0Ksequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_138/dense_138/ActivityRegularizer/strided_slice?
1sequential_138/dense_138/ActivityRegularizer/CastCastCsequential_138/dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_138/dense_138/ActivityRegularizer/Cast?
6sequential_138/dense_138/ActivityRegularizer/truediv_2RealDiv6sequential_138/dense_138/ActivityRegularizer/mul_2:z:05sequential_138/dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_138/dense_138/ActivityRegularizer/truediv_2?
.sequential_139/dense_139/MatMul/ReadVariableOpReadVariableOp7sequential_139_dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_139/dense_139/MatMul/ReadVariableOp?
sequential_139/dense_139/MatMulMatMul$sequential_138/dense_138/Sigmoid:y:06sequential_139/dense_139/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_139/dense_139/MatMul?
/sequential_139/dense_139/BiasAdd/ReadVariableOpReadVariableOp8sequential_139_dense_139_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_139/dense_139/BiasAdd/ReadVariableOp?
 sequential_139/dense_139/BiasAddBiasAdd)sequential_139/dense_139/MatMul:product:07sequential_139/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_139/dense_139/BiasAdd?
 sequential_139/dense_139/SigmoidSigmoid)sequential_139/dense_139/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_139/dense_139/Sigmoid?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_138_dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_139_dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity$sequential_139/dense_139/Sigmoid:y:03^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp0^sequential_138/dense_138/BiasAdd/ReadVariableOp/^sequential_138/dense_138/MatMul/ReadVariableOp0^sequential_139/dense_139/BiasAdd/ReadVariableOp/^sequential_139/dense_139/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_138/dense_138/ActivityRegularizer/truediv_2:z:03^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp0^sequential_138/dense_138/BiasAdd/ReadVariableOp/^sequential_138/dense_138/MatMul/ReadVariableOp0^sequential_139/dense_139/BiasAdd/ReadVariableOp/^sequential_139/dense_139/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_138/dense_138/BiasAdd/ReadVariableOp/sequential_138/dense_138/BiasAdd/ReadVariableOp2`
.sequential_138/dense_138/MatMul/ReadVariableOp.sequential_138/dense_138/MatMul/ReadVariableOp2b
/sequential_139/dense_139/BiasAdd/ReadVariableOp/sequential_139/dense_139/BiasAdd/ReadVariableOp2`
.sequential_139/dense_139/MatMul/ReadVariableOp.sequential_139/dense_139/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
1__inference_sequential_139_layer_call_fn_16662824

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_139_layer_call_and_return_conditional_losses_166622502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_16662536
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_166620062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?#
?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662189
input_70$
dense_138_16662168:^  
dense_138_16662170: 
identity

identity_1??!dense_138/StatefulPartitionedCall?2dense_138/kernel/Regularizer/Square/ReadVariableOp?
!dense_138/StatefulPartitionedCallStatefulPartitionedCallinput_70dense_138_16662168dense_138_16662170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_138_layer_call_and_return_conditional_losses_166620592#
!dense_138/StatefulPartitionedCall?
-dense_138/ActivityRegularizer/PartitionedCallPartitionedCall*dense_138/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *<
f7R5
3__inference_dense_138_activity_regularizer_166620352/
-dense_138/ActivityRegularizer/PartitionedCall?
#dense_138/ActivityRegularizer/ShapeShape*dense_138/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_138/ActivityRegularizer/Shape?
1dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_138/ActivityRegularizer/strided_slice/stack?
3dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_1?
3dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_2?
+dense_138/ActivityRegularizer/strided_sliceStridedSlice,dense_138/ActivityRegularizer/Shape:output:0:dense_138/ActivityRegularizer/strided_slice/stack:output:0<dense_138/ActivityRegularizer/strided_slice/stack_1:output:0<dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_138/ActivityRegularizer/strided_slice?
"dense_138/ActivityRegularizer/CastCast4dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_138/ActivityRegularizer/Cast?
%dense_138/ActivityRegularizer/truedivRealDiv6dense_138/ActivityRegularizer/PartitionedCall:output:0&dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_138/ActivityRegularizer/truediv?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_138_16662168*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentity*dense_138/StatefulPartitionedCall:output:0"^dense_138/StatefulPartitionedCall3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_138/ActivityRegularizer/truediv:z:0"^dense_138/StatefulPartitionedCall3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_138/StatefulPartitionedCall!dense_138/StatefulPartitionedCall2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_70
?
?
$__inference__traced_restore_16663064
file_prefix3
!assignvariableop_dense_138_kernel:^ /
!assignvariableop_1_dense_138_bias: 5
#assignvariableop_2_dense_139_kernel: ^/
!assignvariableop_3_dense_139_bias:^

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_138_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_138_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_139_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_139_biasIdentity_3:output:0"/device:CPU:0*
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
?%
?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662371
x)
sequential_138_16662346:^ %
sequential_138_16662348: )
sequential_139_16662352: ^%
sequential_139_16662354:^
identity

identity_1??2dense_138/kernel/Regularizer/Square/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?&sequential_138/StatefulPartitionedCall?&sequential_139/StatefulPartitionedCall?
&sequential_138/StatefulPartitionedCallStatefulPartitionedCallxsequential_138_16662346sequential_138_16662348*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_138_layer_call_and_return_conditional_losses_166620812(
&sequential_138/StatefulPartitionedCall?
&sequential_139/StatefulPartitionedCallStatefulPartitionedCall/sequential_138/StatefulPartitionedCall:output:0sequential_139_16662352sequential_139_16662354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_139_layer_call_and_return_conditional_losses_166622502(
&sequential_139/StatefulPartitionedCall?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_138_16662346*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_139_16662352*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity/sequential_139/StatefulPartitionedCall:output:03^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp'^sequential_138/StatefulPartitionedCall'^sequential_139/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_138/StatefulPartitionedCall:output:13^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp'^sequential_138/StatefulPartitionedCall'^sequential_139/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_138/StatefulPartitionedCall&sequential_138/StatefulPartitionedCall2P
&sequential_139/StatefulPartitionedCall&sequential_139/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
1__inference_autoencoder_69_layer_call_fn_16662550
x
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_166623712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?_
?
#__inference__wrapped_model_16662006
input_1X
Fautoencoder_69_sequential_138_dense_138_matmul_readvariableop_resource:^ U
Gautoencoder_69_sequential_138_dense_138_biasadd_readvariableop_resource: X
Fautoencoder_69_sequential_139_dense_139_matmul_readvariableop_resource: ^U
Gautoencoder_69_sequential_139_dense_139_biasadd_readvariableop_resource:^
identity??>autoencoder_69/sequential_138/dense_138/BiasAdd/ReadVariableOp?=autoencoder_69/sequential_138/dense_138/MatMul/ReadVariableOp?>autoencoder_69/sequential_139/dense_139/BiasAdd/ReadVariableOp?=autoencoder_69/sequential_139/dense_139/MatMul/ReadVariableOp?
=autoencoder_69/sequential_138/dense_138/MatMul/ReadVariableOpReadVariableOpFautoencoder_69_sequential_138_dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02?
=autoencoder_69/sequential_138/dense_138/MatMul/ReadVariableOp?
.autoencoder_69/sequential_138/dense_138/MatMulMatMulinput_1Eautoencoder_69/sequential_138/dense_138/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 20
.autoencoder_69/sequential_138/dense_138/MatMul?
>autoencoder_69/sequential_138/dense_138/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_69_sequential_138_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>autoencoder_69/sequential_138/dense_138/BiasAdd/ReadVariableOp?
/autoencoder_69/sequential_138/dense_138/BiasAddBiasAdd8autoencoder_69/sequential_138/dense_138/MatMul:product:0Fautoencoder_69/sequential_138/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_69/sequential_138/dense_138/BiasAdd?
/autoencoder_69/sequential_138/dense_138/SigmoidSigmoid8autoencoder_69/sequential_138/dense_138/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_69/sequential_138/dense_138/Sigmoid?
Rautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Mean/reduction_indices?
@autoencoder_69/sequential_138/dense_138/ActivityRegularizer/MeanMean3autoencoder_69/sequential_138/dense_138/Sigmoid:y:0[autoencoder_69/sequential_138/dense_138/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2B
@autoencoder_69/sequential_138/dense_138/ActivityRegularizer/Mean?
Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2G
Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Maximum/y?
Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/MaximumMaximumIautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Mean:output:0Nautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2E
Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Maximum?
Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2G
Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv/x?
Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truedivRealDivNautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv/x:output:0Gautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv?
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/LogLogGautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2A
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/Log?
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2C
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul/x?
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/mulMulJautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul/x:output:0Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2A
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul?
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2C
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/sub/x?
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/subSubJautoencoder_69/sequential_138/dense_138/ActivityRegularizer/sub/x:output:0Gautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2A
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/sub?
Gautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2I
Gautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv_1/x?
Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv_1RealDivPautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv_1/x:output:0Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2G
Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv_1?
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Log_1LogIautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Log_1?
Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2E
Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_1/x?
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_1MulLautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_1/x:output:0Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2C
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_1?
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/addAddV2Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul:z:0Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/add?
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Const?
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/SumSumCautoencoder_69/sequential_138/dense_138/ActivityRegularizer/add:z:0Jautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2A
?autoencoder_69/sequential_138/dense_138/ActivityRegularizer/Sum?
Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2E
Cautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_2/x?
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_2MulLautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_2/x:output:0Hautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_2?
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/ShapeShape3autoencoder_69/sequential_138/dense_138/Sigmoid:y:0*
T0*
_output_shapes
:2C
Aautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Shape?
Oautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stack?
Qautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1?
Qautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2?
Iautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_sliceStridedSliceJautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Shape:output:0Xautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stack:output:0Zautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stack_1:output:0Zautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice?
@autoencoder_69/sequential_138/dense_138/ActivityRegularizer/CastCastRautoencoder_69/sequential_138/dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2B
@autoencoder_69/sequential_138/dense_138/ActivityRegularizer/Cast?
Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv_2RealDivEautoencoder_69/sequential_138/dense_138/ActivityRegularizer/mul_2:z:0Dautoencoder_69/sequential_138/dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2G
Eautoencoder_69/sequential_138/dense_138/ActivityRegularizer/truediv_2?
=autoencoder_69/sequential_139/dense_139/MatMul/ReadVariableOpReadVariableOpFautoencoder_69_sequential_139_dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02?
=autoencoder_69/sequential_139/dense_139/MatMul/ReadVariableOp?
.autoencoder_69/sequential_139/dense_139/MatMulMatMul3autoencoder_69/sequential_138/dense_138/Sigmoid:y:0Eautoencoder_69/sequential_139/dense_139/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^20
.autoencoder_69/sequential_139/dense_139/MatMul?
>autoencoder_69/sequential_139/dense_139/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_69_sequential_139_dense_139_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02@
>autoencoder_69/sequential_139/dense_139/BiasAdd/ReadVariableOp?
/autoencoder_69/sequential_139/dense_139/BiasAddBiasAdd8autoencoder_69/sequential_139/dense_139/MatMul:product:0Fautoencoder_69/sequential_139/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_69/sequential_139/dense_139/BiasAdd?
/autoencoder_69/sequential_139/dense_139/SigmoidSigmoid8autoencoder_69/sequential_139/dense_139/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_69/sequential_139/dense_139/Sigmoid?
IdentityIdentity3autoencoder_69/sequential_139/dense_139/Sigmoid:y:0?^autoencoder_69/sequential_138/dense_138/BiasAdd/ReadVariableOp>^autoencoder_69/sequential_138/dense_138/MatMul/ReadVariableOp?^autoencoder_69/sequential_139/dense_139/BiasAdd/ReadVariableOp>^autoencoder_69/sequential_139/dense_139/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2?
>autoencoder_69/sequential_138/dense_138/BiasAdd/ReadVariableOp>autoencoder_69/sequential_138/dense_138/BiasAdd/ReadVariableOp2~
=autoencoder_69/sequential_138/dense_138/MatMul/ReadVariableOp=autoencoder_69/sequential_138/dense_138/MatMul/ReadVariableOp2?
>autoencoder_69/sequential_139/dense_139/BiasAdd/ReadVariableOp>autoencoder_69/sequential_139/dense_139/BiasAdd/ReadVariableOp2~
=autoencoder_69/sequential_139/dense_139/MatMul/ReadVariableOp=autoencoder_69/sequential_139/dense_139/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
__inference_loss_fn_1_16662990M
;dense_139_kernel_regularizer_square_readvariableop_resource: ^
identity??2dense_139/kernel/Regularizer/Square/ReadVariableOp?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_139_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity$dense_139/kernel/Regularizer/mul:z:03^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp
?
?
1__inference_autoencoder_69_layer_call_fn_16662453
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_166624272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?B
?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662754

inputs:
(dense_138_matmul_readvariableop_resource:^ 7
)dense_138_biasadd_readvariableop_resource: 
identity

identity_1?? dense_138/BiasAdd/ReadVariableOp?dense_138/MatMul/ReadVariableOp?2dense_138/kernel/Regularizer/Square/ReadVariableOp?
dense_138/MatMul/ReadVariableOpReadVariableOp(dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_138/MatMul/ReadVariableOp?
dense_138/MatMulMatMulinputs'dense_138/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_138/MatMul?
 dense_138/BiasAdd/ReadVariableOpReadVariableOp)dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_138/BiasAdd/ReadVariableOp?
dense_138/BiasAddBiasAdddense_138/MatMul:product:0(dense_138/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_138/BiasAdd
dense_138/SigmoidSigmoiddense_138/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_138/Sigmoid?
4dense_138/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_138/ActivityRegularizer/Mean/reduction_indices?
"dense_138/ActivityRegularizer/MeanMeandense_138/Sigmoid:y:0=dense_138/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_138/ActivityRegularizer/Mean?
'dense_138/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_138/ActivityRegularizer/Maximum/y?
%dense_138/ActivityRegularizer/MaximumMaximum+dense_138/ActivityRegularizer/Mean:output:00dense_138/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_138/ActivityRegularizer/Maximum?
'dense_138/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_138/ActivityRegularizer/truediv/x?
%dense_138/ActivityRegularizer/truedivRealDiv0dense_138/ActivityRegularizer/truediv/x:output:0)dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_138/ActivityRegularizer/truediv?
!dense_138/ActivityRegularizer/LogLog)dense_138/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/Log?
#dense_138/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_138/ActivityRegularizer/mul/x?
!dense_138/ActivityRegularizer/mulMul,dense_138/ActivityRegularizer/mul/x:output:0%dense_138/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/mul?
#dense_138/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_138/ActivityRegularizer/sub/x?
!dense_138/ActivityRegularizer/subSub,dense_138/ActivityRegularizer/sub/x:output:0)dense_138/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/sub?
)dense_138/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_138/ActivityRegularizer/truediv_1/x?
'dense_138/ActivityRegularizer/truediv_1RealDiv2dense_138/ActivityRegularizer/truediv_1/x:output:0%dense_138/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_138/ActivityRegularizer/truediv_1?
#dense_138/ActivityRegularizer/Log_1Log+dense_138/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_138/ActivityRegularizer/Log_1?
%dense_138/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_138/ActivityRegularizer/mul_1/x?
#dense_138/ActivityRegularizer/mul_1Mul.dense_138/ActivityRegularizer/mul_1/x:output:0'dense_138/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_138/ActivityRegularizer/mul_1?
!dense_138/ActivityRegularizer/addAddV2%dense_138/ActivityRegularizer/mul:z:0'dense_138/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/add?
#dense_138/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_138/ActivityRegularizer/Const?
!dense_138/ActivityRegularizer/SumSum%dense_138/ActivityRegularizer/add:z:0,dense_138/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_138/ActivityRegularizer/Sum?
%dense_138/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_138/ActivityRegularizer/mul_2/x?
#dense_138/ActivityRegularizer/mul_2Mul.dense_138/ActivityRegularizer/mul_2/x:output:0*dense_138/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_138/ActivityRegularizer/mul_2?
#dense_138/ActivityRegularizer/ShapeShapedense_138/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_138/ActivityRegularizer/Shape?
1dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_138/ActivityRegularizer/strided_slice/stack?
3dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_1?
3dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_2?
+dense_138/ActivityRegularizer/strided_sliceStridedSlice,dense_138/ActivityRegularizer/Shape:output:0:dense_138/ActivityRegularizer/strided_slice/stack:output:0<dense_138/ActivityRegularizer/strided_slice/stack_1:output:0<dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_138/ActivityRegularizer/strided_slice?
"dense_138/ActivityRegularizer/CastCast4dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_138/ActivityRegularizer/Cast?
'dense_138/ActivityRegularizer/truediv_2RealDiv'dense_138/ActivityRegularizer/mul_2:z:0&dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_138/ActivityRegularizer/truediv_2?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_138_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentitydense_138/Sigmoid:y:0!^dense_138/BiasAdd/ReadVariableOp ^dense_138/MatMul/ReadVariableOp3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_138/ActivityRegularizer/truediv_2:z:0!^dense_138/BiasAdd/ReadVariableOp ^dense_138/MatMul/ReadVariableOp3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_138/BiasAdd/ReadVariableOp dense_138/BiasAdd/ReadVariableOp2B
dense_138/MatMul/ReadVariableOpdense_138/MatMul/ReadVariableOp2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662876

inputs:
(dense_139_matmul_readvariableop_resource: ^7
)dense_139_biasadd_readvariableop_resource:^
identity?? dense_139/BiasAdd/ReadVariableOp?dense_139/MatMul/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?
dense_139/MatMul/ReadVariableOpReadVariableOp(dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_139/MatMul/ReadVariableOp?
dense_139/MatMulMatMulinputs'dense_139/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_139/MatMul?
 dense_139/BiasAdd/ReadVariableOpReadVariableOp)dense_139_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_139/BiasAdd/ReadVariableOp?
dense_139/BiasAddBiasAdddense_139/MatMul:product:0(dense_139/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_139/BiasAdd
dense_139/SigmoidSigmoiddense_139/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_139/Sigmoid?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentitydense_139/Sigmoid:y:0!^dense_139/BiasAdd/ReadVariableOp ^dense_139/MatMul/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_139/BiasAdd/ReadVariableOp dense_139/BiasAdd/ReadVariableOp2B
dense_139/MatMul/ReadVariableOpdense_139/MatMul/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662893
dense_139_input:
(dense_139_matmul_readvariableop_resource: ^7
)dense_139_biasadd_readvariableop_resource:^
identity?? dense_139/BiasAdd/ReadVariableOp?dense_139/MatMul/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?
dense_139/MatMul/ReadVariableOpReadVariableOp(dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_139/MatMul/ReadVariableOp?
dense_139/MatMulMatMuldense_139_input'dense_139/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_139/MatMul?
 dense_139/BiasAdd/ReadVariableOpReadVariableOp)dense_139_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_139/BiasAdd/ReadVariableOp?
dense_139/BiasAddBiasAdddense_139/MatMul:product:0(dense_139/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_139/BiasAdd
dense_139/SigmoidSigmoiddense_139/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_139/Sigmoid?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentitydense_139/Sigmoid:y:0!^dense_139/BiasAdd/ReadVariableOp ^dense_139/MatMul/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_139/BiasAdd/ReadVariableOp dense_139/BiasAdd/ReadVariableOp2B
dense_139/MatMul/ReadVariableOpdense_139/MatMul/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_139_input
?#
?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662081

inputs$
dense_138_16662060:^  
dense_138_16662062: 
identity

identity_1??!dense_138/StatefulPartitionedCall?2dense_138/kernel/Regularizer/Square/ReadVariableOp?
!dense_138/StatefulPartitionedCallStatefulPartitionedCallinputsdense_138_16662060dense_138_16662062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_138_layer_call_and_return_conditional_losses_166620592#
!dense_138/StatefulPartitionedCall?
-dense_138/ActivityRegularizer/PartitionedCallPartitionedCall*dense_138/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *<
f7R5
3__inference_dense_138_activity_regularizer_166620352/
-dense_138/ActivityRegularizer/PartitionedCall?
#dense_138/ActivityRegularizer/ShapeShape*dense_138/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_138/ActivityRegularizer/Shape?
1dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_138/ActivityRegularizer/strided_slice/stack?
3dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_1?
3dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_2?
+dense_138/ActivityRegularizer/strided_sliceStridedSlice,dense_138/ActivityRegularizer/Shape:output:0:dense_138/ActivityRegularizer/strided_slice/stack:output:0<dense_138/ActivityRegularizer/strided_slice/stack_1:output:0<dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_138/ActivityRegularizer/strided_slice?
"dense_138/ActivityRegularizer/CastCast4dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_138/ActivityRegularizer/Cast?
%dense_138/ActivityRegularizer/truedivRealDiv6dense_138/ActivityRegularizer/PartitionedCall:output:0&dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_138/ActivityRegularizer/truediv?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_138_16662060*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentity*dense_138/StatefulPartitionedCall:output:0"^dense_138/StatefulPartitionedCall3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_138/ActivityRegularizer/truediv:z:0"^dense_138/StatefulPartitionedCall3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_138/StatefulPartitionedCall!dense_138/StatefulPartitionedCall2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
!__inference__traced_save_16663042
file_prefix/
+savev2_dense_138_kernel_read_readvariableop-
)savev2_dense_138_bias_read_readvariableop/
+savev2_dense_139_kernel_read_readvariableop-
)savev2_dense_139_bias_read_readvariableop
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
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_138_kernel_read_readvariableop)savev2_dense_138_bias_read_readvariableop+savev2_dense_139_kernel_read_readvariableop)savev2_dense_139_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
,__inference_dense_139_layer_call_fn_16662962

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_139_layer_call_and_return_conditional_losses_166622372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?#
?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662213
input_70$
dense_138_16662192:^  
dense_138_16662194: 
identity

identity_1??!dense_138/StatefulPartitionedCall?2dense_138/kernel/Regularizer/Square/ReadVariableOp?
!dense_138/StatefulPartitionedCallStatefulPartitionedCallinput_70dense_138_16662192dense_138_16662194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_138_layer_call_and_return_conditional_losses_166620592#
!dense_138/StatefulPartitionedCall?
-dense_138/ActivityRegularizer/PartitionedCallPartitionedCall*dense_138/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *<
f7R5
3__inference_dense_138_activity_regularizer_166620352/
-dense_138/ActivityRegularizer/PartitionedCall?
#dense_138/ActivityRegularizer/ShapeShape*dense_138/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_138/ActivityRegularizer/Shape?
1dense_138/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_138/ActivityRegularizer/strided_slice/stack?
3dense_138/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_1?
3dense_138/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_138/ActivityRegularizer/strided_slice/stack_2?
+dense_138/ActivityRegularizer/strided_sliceStridedSlice,dense_138/ActivityRegularizer/Shape:output:0:dense_138/ActivityRegularizer/strided_slice/stack:output:0<dense_138/ActivityRegularizer/strided_slice/stack_1:output:0<dense_138/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_138/ActivityRegularizer/strided_slice?
"dense_138/ActivityRegularizer/CastCast4dense_138/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_138/ActivityRegularizer/Cast?
%dense_138/ActivityRegularizer/truedivRealDiv6dense_138/ActivityRegularizer/PartitionedCall:output:0&dense_138/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_138/ActivityRegularizer/truediv?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_138_16662192*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
IdentityIdentity*dense_138/StatefulPartitionedCall:output:0"^dense_138/StatefulPartitionedCall3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_138/ActivityRegularizer/truediv:z:0"^dense_138/StatefulPartitionedCall3^dense_138/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_138/StatefulPartitionedCall!dense_138/StatefulPartitionedCall2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_70
?
?
1__inference_sequential_139_layer_call_fn_16662815
dense_139_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_139_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_139_layer_call_and_return_conditional_losses_166622502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_139_input
?
?
G__inference_dense_139_layer_call_and_return_conditional_losses_16662237

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????^2	
Sigmoid?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_sequential_139_layer_call_fn_16662842
dense_139_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_139_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_139_layer_call_and_return_conditional_losses_166622932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_139_input
?
?
,__inference_dense_138_layer_call_fn_16662925

inputs
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_138_layer_call_and_return_conditional_losses_166620592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662859

inputs:
(dense_139_matmul_readvariableop_resource: ^7
)dense_139_biasadd_readvariableop_resource:^
identity?? dense_139/BiasAdd/ReadVariableOp?dense_139/MatMul/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?
dense_139/MatMul/ReadVariableOpReadVariableOp(dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_139/MatMul/ReadVariableOp?
dense_139/MatMulMatMulinputs'dense_139/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_139/MatMul?
 dense_139/BiasAdd/ReadVariableOpReadVariableOp)dense_139_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_139/BiasAdd/ReadVariableOp?
dense_139/BiasAddBiasAdddense_139/MatMul:product:0(dense_139/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_139/BiasAdd
dense_139/SigmoidSigmoiddense_139/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_139/Sigmoid?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_139_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentitydense_139/Sigmoid:y:0!^dense_139/BiasAdd/ReadVariableOp ^dense_139/MatMul/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_139/BiasAdd/ReadVariableOp dense_139/BiasAdd/ReadVariableOp2B
dense_139/MatMul/ReadVariableOpdense_139/MatMul/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_sequential_138_layer_call_fn_16662165
input_70
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_70unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_138_layer_call_and_return_conditional_losses_166621472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_70
?
?
1__inference_autoencoder_69_layer_call_fn_16662564
x
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_166624272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?%
?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662481
input_1)
sequential_138_16662456:^ %
sequential_138_16662458: )
sequential_139_16662462: ^%
sequential_139_16662464:^
identity

identity_1??2dense_138/kernel/Regularizer/Square/ReadVariableOp?2dense_139/kernel/Regularizer/Square/ReadVariableOp?&sequential_138/StatefulPartitionedCall?&sequential_139/StatefulPartitionedCall?
&sequential_138/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_138_16662456sequential_138_16662458*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_138_layer_call_and_return_conditional_losses_166620812(
&sequential_138/StatefulPartitionedCall?
&sequential_139/StatefulPartitionedCallStatefulPartitionedCall/sequential_138/StatefulPartitionedCall:output:0sequential_139_16662462sequential_139_16662464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_139_layer_call_and_return_conditional_losses_166622502(
&sequential_139/StatefulPartitionedCall?
2dense_138/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_138_16662456*
_output_shapes

:^ *
dtype024
2dense_138/kernel/Regularizer/Square/ReadVariableOp?
#dense_138/kernel/Regularizer/SquareSquare:dense_138/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_138/kernel/Regularizer/Square?
"dense_138/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_138/kernel/Regularizer/Const?
 dense_138/kernel/Regularizer/SumSum'dense_138/kernel/Regularizer/Square:y:0+dense_138/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/Sum?
"dense_138/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_138/kernel/Regularizer/mul/x?
 dense_138/kernel/Regularizer/mulMul+dense_138/kernel/Regularizer/mul/x:output:0)dense_138/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_138/kernel/Regularizer/mul?
2dense_139/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_139_16662462*
_output_shapes

: ^*
dtype024
2dense_139/kernel/Regularizer/Square/ReadVariableOp?
#dense_139/kernel/Regularizer/SquareSquare:dense_139/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_139/kernel/Regularizer/Square?
"dense_139/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_139/kernel/Regularizer/Const?
 dense_139/kernel/Regularizer/SumSum'dense_139/kernel/Regularizer/Square:y:0+dense_139/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/Sum?
"dense_139/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_139/kernel/Regularizer/mul/x?
 dense_139/kernel/Regularizer/mulMul+dense_139/kernel/Regularizer/mul/x:output:0)dense_139/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_139/kernel/Regularizer/mul?
IdentityIdentity/sequential_139/StatefulPartitionedCall:output:03^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp'^sequential_138/StatefulPartitionedCall'^sequential_139/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_138/StatefulPartitionedCall:output:13^dense_138/kernel/Regularizer/Square/ReadVariableOp3^dense_139/kernel/Regularizer/Square/ReadVariableOp'^sequential_138/StatefulPartitionedCall'^sequential_139/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_138/kernel/Regularizer/Square/ReadVariableOp2dense_138/kernel/Regularizer/Square/ReadVariableOp2h
2dense_139/kernel/Regularizer/Square/ReadVariableOp2dense_139/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_138/StatefulPartitionedCall&sequential_138/StatefulPartitionedCall2P
&sequential_139/StatefulPartitionedCall&sequential_139/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????^<
output_10
StatefulPartitionedCall:0?????????^tensorflow/serving/predict:??
?
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
*:&call_and_return_all_conditional_losses"?
_tf_keras_model?{"name": "autoencoder_69", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
 "
trackable_dict_wrapper
?
	layer_with_weights-0
	layer-0

trainable_variables
	variables
regularization_losses
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_138", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_138", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_70"}}, {"class_name": "Dense", "config": {"name": "dense_138", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_70"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_138", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_70"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_138", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_139", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_139", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_139_input"}}, {"class_name": "Dense", "config": {"name": "dense_139", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_139_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_139", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_139_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_139", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "dense_138", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_138", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
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
?	

kernel
bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
C__call__
*D&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_139", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_139", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
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
": ^ 2dense_138/kernel
: 2dense_138/bias
":  ^2dense_139/kernel
:^2dense_139/bias
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
?
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
?2?
1__inference_autoencoder_69_layer_call_fn_16662383
1__inference_autoencoder_69_layer_call_fn_16662550
1__inference_autoencoder_69_layer_call_fn_16662564
1__inference_autoencoder_69_layer_call_fn_16662453?
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
?2?
#__inference__wrapped_model_16662006?
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
annotations? *&?#
!?
input_1?????????^
?2?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662623
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662682
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662481
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662509?
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
1__inference_sequential_138_layer_call_fn_16662089
1__inference_sequential_138_layer_call_fn_16662698
1__inference_sequential_138_layer_call_fn_16662708
1__inference_sequential_138_layer_call_fn_16662165?
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
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662754
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662800
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662189
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662213?
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
1__inference_sequential_139_layer_call_fn_16662815
1__inference_sequential_139_layer_call_fn_16662824
1__inference_sequential_139_layer_call_fn_16662833
1__inference_sequential_139_layer_call_fn_16662842?
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
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662859
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662876
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662893
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662910?
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
&__inference_signature_wrapper_16662536input_1"?
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
,__inference_dense_138_layer_call_fn_16662925?
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
K__inference_dense_138_layer_call_and_return_all_conditional_losses_16662936?
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
__inference_loss_fn_0_16662947?
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
,__inference_dense_139_layer_call_fn_16662962?
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
G__inference_dense_139_layer_call_and_return_conditional_losses_16662979?
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
__inference_loss_fn_1_16662990?
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
3__inference_dense_138_activity_regularizer_16662035?
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
G__inference_dense_138_layer_call_and_return_conditional_losses_16663007?
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
#__inference__wrapped_model_16662006m0?-
&?#
!?
input_1?????????^
? "3?0
.
output_1"?
output_1?????????^?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662481q4?1
*?'
!?
input_1?????????^
p 
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662509q4?1
*?'
!?
input_1?????????^
p
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662623k.?+
$?!
?
X?????????^
p 
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_69_layer_call_and_return_conditional_losses_16662682k.?+
$?!
?
X?????????^
p
? "3?0
?
0?????????^
?
?	
1/0 ?
1__inference_autoencoder_69_layer_call_fn_16662383V4?1
*?'
!?
input_1?????????^
p 
? "??????????^?
1__inference_autoencoder_69_layer_call_fn_16662453V4?1
*?'
!?
input_1?????????^
p
? "??????????^?
1__inference_autoencoder_69_layer_call_fn_16662550P.?+
$?!
?
X?????????^
p 
? "??????????^?
1__inference_autoencoder_69_layer_call_fn_16662564P.?+
$?!
?
X?????????^
p
? "??????????^f
3__inference_dense_138_activity_regularizer_16662035/$?!
?
?

activation
? "? ?
K__inference_dense_138_layer_call_and_return_all_conditional_losses_16662936j/?,
%?"
 ?
inputs?????????^
? "3?0
?
0????????? 
?
?	
1/0 ?
G__inference_dense_138_layer_call_and_return_conditional_losses_16663007\/?,
%?"
 ?
inputs?????????^
? "%?"
?
0????????? 
? 
,__inference_dense_138_layer_call_fn_16662925O/?,
%?"
 ?
inputs?????????^
? "?????????? ?
G__inference_dense_139_layer_call_and_return_conditional_losses_16662979\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????^
? 
,__inference_dense_139_layer_call_fn_16662962O/?,
%?"
 ?
inputs????????? 
? "??????????^=
__inference_loss_fn_0_16662947?

? 
? "? =
__inference_loss_fn_1_16662990?

? 
? "? ?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662189t9?6
/?,
"?
input_70?????????^
p 

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662213t9?6
/?,
"?
input_70?????????^
p

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662754r7?4
-?*
 ?
inputs?????????^
p 

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_138_layer_call_and_return_conditional_losses_16662800r7?4
-?*
 ?
inputs?????????^
p

 
? "3?0
?
0????????? 
?
?	
1/0 ?
1__inference_sequential_138_layer_call_fn_16662089Y9?6
/?,
"?
input_70?????????^
p 

 
? "?????????? ?
1__inference_sequential_138_layer_call_fn_16662165Y9?6
/?,
"?
input_70?????????^
p

 
? "?????????? ?
1__inference_sequential_138_layer_call_fn_16662698W7?4
-?*
 ?
inputs?????????^
p 

 
? "?????????? ?
1__inference_sequential_138_layer_call_fn_16662708W7?4
-?*
 ?
inputs?????????^
p

 
? "?????????? ?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662859d7?4
-?*
 ?
inputs????????? 
p 

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662876d7?4
-?*
 ?
inputs????????? 
p

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662893m@?=
6?3
)?&
dense_139_input????????? 
p 

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_139_layer_call_and_return_conditional_losses_16662910m@?=
6?3
)?&
dense_139_input????????? 
p

 
? "%?"
?
0?????????^
? ?
1__inference_sequential_139_layer_call_fn_16662815`@?=
6?3
)?&
dense_139_input????????? 
p 

 
? "??????????^?
1__inference_sequential_139_layer_call_fn_16662824W7?4
-?*
 ?
inputs????????? 
p 

 
? "??????????^?
1__inference_sequential_139_layer_call_fn_16662833W7?4
-?*
 ?
inputs????????? 
p

 
? "??????????^?
1__inference_sequential_139_layer_call_fn_16662842`@?=
6?3
)?&
dense_139_input????????? 
p

 
? "??????????^?
&__inference_signature_wrapper_16662536x;?8
? 
1?.
,
input_1!?
input_1?????????^"3?0
.
output_1"?
output_1?????????^