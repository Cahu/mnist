use std::slice;
use std::io::Cursor;
use std::io::prelude::*;
use std::fs::File;

use byteorder::{BigEndian, ReadBytesExt};

use mnisterror::Result;

const MAGIC_NUMBER: u32 = 0x801;

pub struct Labels {
    num_labels:  usize,
    data:        Vec<u8>,
}

pub struct Iter<'a>(slice::Iter<'a, u8>);

impl Labels {
    pub fn new(path: &str) -> Result<Self> {

        let mut f = File::open(path)
            .map_err(|e| format_err!("Could not open the file: {}", e))?;

        let mut header = [0u8; 8];
        match f.read(&mut header) {
            Ok(8)  => {}
            Ok(s)  => { return Err(format_err!("Bad header size: {}, expected 8", s)) }
            Err(e) => { return Err(format_err!("Could not read header: {}", e))  }
        }

        let mut cursor = Cursor::new(header);

        // verify magic number in header
        match cursor.read_u32::<BigEndian>() {
            Ok(MAGIC_NUMBER) => {}
            _                => {
                return Err(format_err!("Could not find the magic number ({:#X}) for labels files", MAGIC_NUMBER))
            }
        }

        let num_labels = cursor.read_u32::<BigEndian>()
            .map(|e| e as usize)
            .map_err(|err| format_err!("Error while extracting number of labels: {}", err))?;

        let mut data = Vec::with_capacity(num_labels);
        match f.read_to_end(&mut data) {
            Ok(s) if s == num_labels => {}
            Ok(s)  => { return Err(format_err!("Incorrect size: {} bytes, expected {} as specified in the header", s, num_labels)) }      
            Err(e) => { return Err(format_err!("Could not read labels: {}", e))  }      
        }

        Ok(Labels {
            num_labels: num_labels,
            data:       data,
        })
    }

    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    pub fn iter(&self) -> Iter {
        Iter(self.data.iter())
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a u8;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

