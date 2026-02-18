import { NextResponse } from 'next/server';

export function middleware(request) {
  const { pathname } = request.nextUrl;

  // /api-ref or /api-ref/ → serve index.html
  if (pathname === '/api-ref' || pathname === '/api-ref/') {
    const url = request.nextUrl.clone();
    url.pathname = '/api-ref/index.html';
    return NextResponse.rewrite(url);
  }

  // /api-ref/foo (no file extension) → serve foo.html
  if (pathname.startsWith('/api-ref/') && !pathname.match(/\.\w+$/)) {
    const url = request.nextUrl.clone();
    url.pathname = pathname.replace(/\/$/, '') + '.html';
    return NextResponse.rewrite(url);
  }
}

export const config = {
  matcher: ['/api-ref', '/api-ref/:path*'],
};
