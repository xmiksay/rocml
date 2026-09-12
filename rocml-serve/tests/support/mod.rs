//! A deliberately minimal hand-rolled HTTP/1.1 client for the end-to-end
//! test: the workspace's dependency allow-list for `rocml-serve` doesn't
//! include an HTTP client crate (`reqwest`, `hyper` as a client, etc.), and
//! adding one just for tests would be exactly the "extras" the brief asks
//! to avoid. We only need to POST JSON and read back either a
//! `Content-Length` body or a chunked (SSE) one, both well-formed because
//! we control the server on the other end — so a ~60-line client is enough,
//! and it exercises the real TCP/HTTP path `axum::serve` runs in production.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

/// Sends `POST {path}` with a JSON body to `addr` and reads the response to
/// completion. Always sends `Connection: close` so the server closes the
/// socket once it's done, which is what lets `read_to_end` return instead of
/// blocking forever waiting for more bytes on a keep-alive connection.
pub async fn post_json(addr: std::net::SocketAddr, path: &str, body: &str) -> HttpResponse {
    let mut stream = TcpStream::connect(addr)
        .await
        .expect("connect to test server");
    let request = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    parse_response(&raw)
}

/// Sends `GET {path}` to `addr` and reads the response to completion — the
/// `POST`-only counterpart above, for issue #9's `GET /debug/last_prompt`.
pub async fn get(addr: std::net::SocketAddr, path: &str) -> HttpResponse {
    let mut stream = TcpStream::connect(addr)
        .await
        .expect("connect to test server");
    let request = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\
         \r\n"
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> HttpResponse {
    let header_end = find(raw, b"\r\n\r\n").expect("response has no header/body separator") + 4;
    let header_text = String::from_utf8_lossy(&raw[..header_end]);
    let status = header_text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);

    let raw_body = &raw[header_end..];
    let is_chunked = header_text
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked");
    let body_bytes = if is_chunked {
        dechunk(raw_body)
    } else {
        raw_body.to_vec()
    };
    HttpResponse {
        status,
        body: String::from_utf8_lossy(&body_bytes).into_owned(),
    }
}

/// Undoes HTTP/1.1 chunked transfer encoding: each chunk is a hex length
/// line, CRLF, that many body bytes, CRLF, repeated until a zero-length
/// chunk. Stops (rather than erroring) on anything unexpected, since a test
/// helper has no one to report a parse error to but the assertion that
/// follows it.
fn dechunk(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let Some(line_len) = find(&data[pos..], b"\r\n") else {
            break;
        };
        let size_line = String::from_utf8_lossy(&data[pos..pos + line_len]);
        let Ok(size) = usize::from_str_radix(size_line.trim(), 16) else {
            break;
        };
        pos += line_len + 2;
        if size == 0 || pos + size > data.len() {
            break;
        }
        out.extend_from_slice(&data[pos..pos + size]);
        pos += size + 2; // skip the chunk's trailing CRLF
    }
    out
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
