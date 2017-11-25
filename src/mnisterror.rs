use std::result;
use failure;

pub type Result<T> = result::Result<T, failure::Error>;
