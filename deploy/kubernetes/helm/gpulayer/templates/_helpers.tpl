{{- define "gpulayer.fullname" -}}
{{- printf "%s-gpulayer" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "gpulayer.labels" -}}
app.kubernetes.io/name: gpulayer
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "gpulayer.serverHost" -}}
{{- printf "%s-server.%s.svc" (include "gpulayer.fullname" .) .Release.Namespace -}}
{{- end -}}

{{- define "gpulayer.image" -}}
{{- $img := .img | default dict -}}
{{- $repo := $img.repository | default .root.Values.image.repository -}}
{{- $base := .root.Values.image.tag | default .root.Chart.AppVersion -}}
{{- $tag := $img.tag | default (printf "%s%s" $base (ternary (printf "-%s" ($img.variant | default "")) "" (not (empty $img.variant)))) -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}

{{- define "gpulayer.tokenSecret" -}}
{{- .Values.existingTokenSecret | default (printf "%s-tokens" (include "gpulayer.fullname" .)) -}}
{{- end -}}
