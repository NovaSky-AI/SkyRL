import { NextResponse } from 'next/server';

export function middleware(request) {
  const { pathname } = request.nextUrl;

  // /api-ref without trailing slash → redirect to /api-ref/
  if (pathname === '/api-ref') {
    const url = request.nextUrl.clone();
    url.pathname = '/api-ref/';
    return NextResponse.redirect(url, 308);
  }

  // /api-ref/ or /api-ref/foo/ → serve index.html
  if (pathname.startsWith('/api-ref/') && !pathname.match(/\.\w+$/)) {
    const url = request.nextUrl.clone();
    url.pathname = pathname.endsWith('/')
      ? pathname + 'index.html'
      : pathname + '/index.html';
    return NextResponse.rewrite(url);
  }
}

export const config = {
  matcher: ['/api-ref', '/api-ref/:path*'],
};
