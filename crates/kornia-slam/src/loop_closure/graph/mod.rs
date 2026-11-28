//! Gates a verified loop passes before it may enter the pose graph.

mod gate;

pub(crate) use gate::{
    GateDecision, LoopGate, LoopProposal, OdometryConsistencyGate, TemporalConsistencyGate,
};
