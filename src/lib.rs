//! Rules:
//! We have `k` drones at our disposal to reconnoitre a fixed number
//! of `n` sites `A_0, ..., A_(n - 1)`.
//! Associated with each site is a success propability `p_0, ..., p_(n - 1)`.
//! If a drone does not succeed in the reconnaisance of a site, the drone is lost.
//! During their mission, a drone can send, but not recieve information.
//! Thus, a site is successfully reconnoitred, if at least one drone succeeded,
//! regardless if the drone is lost afterwards.
//! Under these conditions and with all drones beeing send at once,
//! what is the ideal order in which each drone visits each site?
//! We assume no losses while the drones are approaching a site.

mod prob;
pub use prob::Prob;

pub mod script;
