/**
 * Original illustrations for the landing page. Flat colour, thick black outline, round caps - the
 * sticker style of the reference, drawn for Penumbra rather than borrowed.
 *
 *   Eclipse     the mascot. A penumbra is the half-shadow between full light and full shadow, so
 *               the character is a sun with a shadow crossing it, arms up, in sneakers.
 *   Packet      a network flow as a little cloud on legs, running.
 *   Binoculars  "see the unseen".
 */

const INK = "#0b0b0b";

export function Sparkle({ className = "", size = 22, color = "#ffffff" }: { className?: string; size?: number; color?: string }) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path
        d="M12 1.5c.6 4.9 2.2 6.9 7 8-4.8 1.1-6.4 3.1-7 8-.6-4.9-2.2-6.9-7-8 4.8-1.1 6.4-3.1 7-8z"
        fill={color}
        stroke={INK}
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Eclipse({ className = "" }: { className?: string }) {
  const rays = Array.from({ length: 12 }, (_, i) => i * 30);
  return (
    <svg className={className} viewBox="0 0 260 250" aria-label="Penumbra mascot: a smiling eclipse">
      <defs>
        <clipPath id="ecl-disk">
          <circle cx="130" cy="100" r="46" />
        </clipPath>
      </defs>
      {/* rays */}
      <g stroke={INK} strokeWidth="4" strokeLinejoin="round">
        {rays.map((deg) => (
          <path
            key={deg}
            d="M130 38 l-9 -20 q9 -6 18 0 z"
            fill="#FFD60A"
            transform={`rotate(${deg} 130 100)`}
          />
        ))}
      </g>
      {/* arms, raised */}
      <g stroke={INK} strokeWidth="5" strokeLinecap="round" fill="none">
        <path d="M90 118 q-26 -8 -42 -40" />
        <path d="M170 118 q26 -8 42 -40" />
      </g>
      <g fill="#ffffff" stroke={INK} strokeWidth="4">
        <circle cx="46" cy="74" r="10" />
        <circle cx="214" cy="74" r="10" />
      </g>
      {/* legs */}
      <g stroke={INK} strokeWidth="5" strokeLinecap="round" fill="none">
        <path d="M112 142 q-4 30 -20 52" />
        <path d="M148 142 q6 30 22 50" />
      </g>
      {/* sneakers */}
      <g stroke={INK} strokeWidth="4" strokeLinejoin="round">
        <path d="M70 196 q2 -12 22 -8 l10 6 q4 10 -4 12 h-26 q-4 0 -2 -10z" fill="#1E6FD9" />
        <path d="M160 190 q10 -6 22 -2 l12 8 q2 10 -6 10 h-26 q-6 -2 -2 -16z" fill="#1E6FD9" />
        <path d="M72 204 h30 M162 204 h32" stroke="#ffffff" strokeWidth="3" />
      </g>
      {/* the disk: light, with the shadow crossing it */}
      <circle cx="130" cy="100" r="46" fill="#FFD60A" />
      <g clipPath="url(#ecl-disk)">
        <circle cx="172" cy="84" r="44" fill="#2b2b40" />
        <circle cx="172" cy="84" r="44" fill="none" stroke="#6d6df0" strokeWidth="10" opacity="0.55" />
      </g>
      <circle cx="130" cy="100" r="46" fill="none" stroke={INK} strokeWidth="5" />
      {/* face, on the lit side */}
      <g stroke={INK} strokeWidth="4" strokeLinecap="round" fill="none">
        <path d="M104 96 q6 -7 12 0" />
        <path d="M128 92 q5 -6 10 0" />
        <path d="M106 112 q13 14 28 0" />
      </g>
      <circle cx="100" cy="108" r="5" fill="#ff7aa8" />
      <circle cx="140" cy="104" r="4" fill="#ff7aa8" />
    </svg>
  );
}

export function Packet({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 220 190" aria-label="A network flow, running">
      {/* motion lines */}
      <g stroke={INK} strokeWidth="4" strokeLinecap="round">
        <path d="M14 70 h26" />
        <path d="M6 92 h30" />
        <path d="M18 114 h20" />
      </g>
      {/* legs + shoes */}
      <g stroke={INK} strokeWidth="5" strokeLinecap="round" fill="none">
        <path d="M92 128 q-8 22 -26 30" />
        <path d="M132 128 q14 18 10 38" />
      </g>
      <g stroke={INK} strokeWidth="4" strokeLinejoin="round" fill="#F02D2D">
        <path d="M44 152 q10 -10 26 -2 l4 10 h-30 q-6 -2 0 -8z" />
        <path d="M130 164 q12 -8 26 0 l2 10 h-30 q-4 -4 2 -10z" />
      </g>
      {/* cloud body */}
      <path
        d="M60 110 q-22 -2 -20 -24 q2 -20 24 -20 q4 -26 32 -26 q22 0 30 20 q10 -10 26 -4 q20 8 14 30 q20 6 14 24 q-4 12 -20 12 h-100z"
        fill="#ffffff"
        stroke={INK}
        strokeWidth="5"
        strokeLinejoin="round"
      />
      {/* arm swinging */}
      <path d="M156 96 q18 6 24 -10" stroke={INK} strokeWidth="5" strokeLinecap="round" fill="none" />
      {/* face */}
      <g stroke={INK} strokeWidth="4" strokeLinecap="round" fill="none">
        <path d="M98 78 v8" />
        <path d="M122 78 v8" />
        <path d="M100 96 q10 10 22 0" />
      </g>
      <text x="92" y="68" fontFamily="var(--font-body), sans-serif" fontWeight="900" fontSize="11" fill={INK}>
        0101
      </text>
    </svg>
  );
}

export function Binoculars({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 240 150" aria-label="Binoculars">
      <g stroke={INK} strokeWidth="5" strokeLinejoin="round" strokeLinecap="round">
        <rect x="96" y="44" width="48" height="30" rx="8" fill="#ffffff" />
        <path d="M40 58 l30 -26 h30 l8 32 z" fill="#ffffff" />
        <path d="M200 58 l-30 -26 h-30 l-8 32 z" fill="#ffffff" />
        <circle cx="68" cy="94" r="42" fill="#ffffff" />
        <circle cx="172" cy="94" r="42" fill="#ffffff" />
        <circle cx="68" cy="94" r="26" fill="#6d6df0" />
        <circle cx="172" cy="94" r="26" fill="#6d6df0" />
      </g>
      <g fill="#ffffff">
        <circle cx="58" cy="84" r="8" />
        <circle cx="162" cy="84" r="8" />
      </g>
      <g stroke={INK} strokeWidth="4" strokeLinecap="round">
        <path d="M20 30 l12 10" />
        <path d="M220 30 l-12 10" />
        <path d="M120 14 v14" />
      </g>
    </svg>
  );
}

export function Target({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 100 100" aria-hidden>
      <g stroke={INK} strokeWidth="4">
        <circle cx="50" cy="50" r="40" fill="#ffffff" />
        <circle cx="50" cy="50" r="27" fill="#F02D2D" />
        <circle cx="50" cy="50" r="13" fill="#ffffff" />
      </g>
      <path d="M50 50 L86 14" stroke={INK} strokeWidth="5" strokeLinecap="round" />
      <path d="M78 10 l10 -4 -4 10" fill="#FFD60A" stroke={INK} strokeWidth="3" strokeLinejoin="round" />
    </svg>
  );
}

export function QuestionEye({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 100 100" aria-hidden>
      <path d="M8 50 q42 -44 84 0 q-42 44 -84 0z" fill="#ffffff" stroke={INK} strokeWidth="4" strokeLinejoin="round" />
      <circle cx="50" cy="50" r="18" fill="#6d6df0" stroke={INK} strokeWidth="4" />
      <text x="50" y="58" textAnchor="middle" fontFamily="var(--font-display), sans-serif" fontSize="24" fill="#ffffff">
        ?
      </text>
    </svg>
  );
}

/** Circular text that turns slowly, like a sticker badge. */
export function Badge({ text, className = "" }: { text: string; className?: string }) {
  return (
    <svg className={className} viewBox="0 0 200 200" aria-label={text}>
      <defs>
        <path id="badge-circle" d="M100 100 m-78 0 a78 78 0 1 1 156 0 a78 78 0 1 1 -156 0" />
      </defs>
      <text fontFamily="var(--font-display), sans-serif" fontSize="23" letterSpacing="2.5" fill="#ffffff">
        <textPath href="#badge-circle">{text}</textPath>
      </text>
    </svg>
  );
}

export function LogoMark({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 40 40" aria-hidden>
      <circle cx="20" cy="20" r="19" fill="#F02D2D" />
      <circle cx="17" cy="20" r="10" fill="#FFD60A" stroke={INK} strokeWidth="2.5" />
      <path d="M22.5 11.6 a10 10 0 0 1 0 16.8 a12 12 0 0 0 0 -16.8z" fill={INK} />
    </svg>
  );
}

export function GitHubMark({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden>
      <path
        fill="currentColor"
        d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.78 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.7 5.4-5.27 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z"
      />
    </svg>
  );
}
