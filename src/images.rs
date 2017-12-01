use std::io::Cursor;
use std::io::prelude::*;
use std::fs::File;

use byteorder::{BigEndian, ReadBytesExt};

use mnisterror::Result;

const MAGIC_NUMBER: u32 = 0x803;

pub struct Images {
    num_images:  usize,
    width:       usize,
    height:      usize,
    data:        Vec<u8>,
}

pub struct Image<'a>{
    data  : &'a [u8],
    width : usize,
    height: usize
}

pub struct Iter<'a> {
    images:   &'a Images,
    cursor:   usize,
}

impl Images {
    pub fn new(path: &str) -> Result<Self> {

        let mut f = File::open(path)
            .map_err(|e| format_err!("Error while opening '{}': {}", path, e))?;

        let mut header = [0u8; 16];
        match f.read(&mut header) {
            Ok(16) => {}
            Ok(s)  => { return Err(format_err!("Bad header size: {}, expected 16", s)) }
            Err(e) => { return Err(format_err!("Could not read header: {}", e))  }
        }

        let mut cursor = Cursor::new(header);

        // verify magic number in header
        match cursor.read_u32::<BigEndian>() {
            Ok(MAGIC_NUMBER) => {}
            _                => {
                return Err(format_err!("Could not find the magic number ({:#X}) for images files", MAGIC_NUMBER))
            }
        }

        let num_images = cursor.read_u32::<BigEndian>()
            .map(|e| e as usize)
            .map_err(|err| format_err!("Error while extracting number of images: {}", err))?;

        let height = cursor.read_u32::<BigEndian>()
            .map(|e| e as usize)
            .map_err(|err| format_err!("Error while extracting the 'height' value: {}", err))?;

        let width = cursor.read_u32::<BigEndian>()
            .map(|e| e as usize)
            .map_err(|err| format_err!("Error while extracting the 'width' value: {}", err))?;

        let size = num_images * height * width;

        let mut data = Vec::with_capacity(size);
        match f.read_to_end(&mut data) {
            Ok(s) if s == size => {}
            Ok(s)  => { return Err(format_err!("Incorrect size: {} bytes, expected {} as specified in the header", s, size)) }      
            Err(e) => { return Err(format_err!("Could not read labels: {}", e))  }      
        }

        Ok(Images {
            num_images: num_images,
            width:      width,
            height:     height,
            data:       data,
        })
    }

    pub fn num_images(&self) -> usize {
        self.num_images
    }

    pub fn image_width(&self) -> usize {
        self.width
    }

    pub fn image_height(&self) -> usize {
        self.height
    }

    pub fn image_size(&self) -> usize {
        self.height * self.width
    }

    pub fn iter(&self) -> Iter {
        Iter {
            cursor: 0,
            images: self,
        }
    }
}

impl<'a> Image<'a> {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn size(&self) -> usize {
        self.width * self.height
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Image<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.images.data.len() {
            return None;
        }

        let size  = self.images.image_size();
        let start = self.cursor;
        self.cursor += size;

        Some(Image {
            data  : &self.images.data[start .. self.cursor],
            width : self.images.width,
            height: self.images.height
        })
    }
}
